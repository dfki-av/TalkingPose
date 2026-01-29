import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime

import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from src.dataset.dance_image import HumanDanceDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.utils.util import delete_additional_ckpt, import_filename, seed_everything

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

LATENT_SCALE = 0.18215


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer: ReferenceAttentionControl,
        reference_control_reader: ReferenceAttentionControl,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):
        pose_fea = self.pose_guider(pose_img.to(device="cuda"))

        if not uncond_fwd:
            ref_ts = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_ts,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        return self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample


def compute_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    alpha = (alphas_cumprod**0.5).to(device=timesteps.device)[timesteps].float()
    sigma = ((1.0 - alphas_cumprod) ** 0.5).to(device=timesteps.device)[timesteps].float()

    while len(alpha.shape) < len(timesteps.shape):
        alpha = alpha[..., None]
    while len(sigma.shape) < len(timesteps.shape):
        sigma = sigma[..., None]

    alpha = alpha.expand(timesteps.shape)
    sigma = sigma.expand(timesteps.shape)
    return (alpha / sigma) ** 2


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(save_dir) if d.startswith(prefix)],
            key=lambda x: int(x.split("-")[1].split(".")[0]),
        )
        if len(checkpoints) >= total_limit:
            n_remove = len(checkpoints) - total_limit + 1
            for name in checkpoints[:n_remove]:
                os.remove(os.path.join(save_dir, name))

    torch.save(model.state_dict(), save_path)


def _make_schedulers(cfg):
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )

    val_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_scheduler = DDIMScheduler(**sched_kwargs)
    return train_scheduler, val_scheduler


def _weight_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Do not support weight dtype: {name} during training")


def _maybe_enable_xformers(cfg, reference_unet, denoising_unet):
    if not cfg.solver.enable_xformers_memory_efficient_attention:
        return
    if not is_xformers_available():
        raise ValueError("xformers is not available. Make sure it is installed correctly")
    reference_unet.enable_xformers_memory_efficient_attention()
    denoising_unet.enable_xformers_memory_efficient_attention()


def _maybe_enable_checkpointing(cfg, reference_unet, denoising_unet):
    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()


def _build_pose_guider(cfg):
    if not cfg.pose_guider_pretrain:
        return PoseGuider(conditioning_embedding_channels=320).to(device="cuda")

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    ).to(device="cuda")

    sd = torch.load(cfg.controlnet_openpose_path)
    to_load = {}
    for k, v in sd.items():
        if k.startswith("controlnet_cond_embedding.") and "conv_out" not in k:
            to_load[k.replace("controlnet_cond_embedding.", "")] = v

    miss, _ = pose_guider.load_state_dict(to_load, strict=False)
    logger.info(f"Missing key for pose guider: {len(miss)}")
    return pose_guider


def _set_trainable(reference_unet, denoising_unet, pose_guider, vae, image_enc):
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    denoising_unet.requires_grad_(True)
    pose_guider.requires_grad_(True)

    for name, param in reference_unet.named_parameters():
        param.requires_grad_(False if "up_blocks.3" in name else True)


def _compute_lr(cfg, accelerator):
    if not cfg.solver.scale_lr:
        return cfg.solver.learning_rate
    return (
        cfg.solver.learning_rate
        * cfg.solver.gradient_accumulation_steps
        * cfg.data.train_bs
        * accelerator.num_processes
    )


def _optimizer_cls(cfg):
    if not cfg.solver.use_8bit_adam:
        return torch.optim.AdamW
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
        ) from e
    return bnb.optim.AdamW8bit


def main(cfg):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    save_dir = f"{cfg.output_dir}/{cfg.exp_name}"
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    weight_dtype = _weight_dtype(cfg.weight_dtype)
    train_noise_scheduler, _ = _make_schedulers(cfg)

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = _build_pose_guider(cfg)
    _set_trainable(reference_unet, denoising_unet, pose_guider, vae, image_enc)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    _maybe_enable_xformers(cfg, reference_unet, denoising_unet)
    _maybe_enable_checkpointing(cfg, reference_unet, denoising_unet)

    lr = _compute_lr(cfg, accelerator)
    opt_cls = _optimizer_cls(cfg)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = opt_cls(
        trainable_params,
        lr=lr,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = HumanDanceDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        img_scale=(0.9, 1.0),
        data_meta_paths=cfg.data.meta_paths,
        sample_margin=cfg.data.sample_margin,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=4,
    )

    net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(cfg.solver.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(cfg.exp_name, init_kwargs={"mlflow": {"run_name": run_time}})
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if cfg.resume_from_checkpoint:
        resume_dir = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint != "latest" else save_dir
        dirs = sorted(
            [d for d in os.listdir(resume_dir) if d.startswith("checkpoint")],
            key=lambda x: int(x.split("-")[1]),
        )
        if dirs:
            ckpt = dirs[-1]
            accelerator.load_state(os.path.join(resume_dir, ckpt))
            accelerator.print(f"Resuming from checkpoint {ckpt}")

            global_step = int(ckpt.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if epoch == first_epoch and step < resume_step:
                continue

            with accelerator.accumulate(net):
                pixel_values = batch["img"].to(weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2) * LATENT_SCALE

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise = noise + cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                tgt_pose_img = batch["tgt_pose"].unsqueeze(2)

                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_imgs = []
                ref_imgs = []
                for ref_img, clip_img in zip(batch["ref_img"], batch["clip_images"]):
                    clip_imgs.append(torch.zeros_like(clip_img) if uncond_fwd else clip_img)
                    ref_imgs.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_imgs, dim=0).to(dtype=vae.dtype, device=vae.device)
                    ref_image_latents = vae.encode(ref_img).latent_dist.sample() * LATENT_SCALE

                    clip_img = torch.stack(clip_imgs, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_embeds = image_enc(clip_img.to("cuda", dtype=weight_dtype)).image_embeds
                    image_prompt_embeds = clip_embeds.unsqueeze(1)

                noisy_latents = train_noise_scheduler.add_noise(latents, noise, timesteps)

                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {train_noise_scheduler.prediction_type}")

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    tgt_pose_img,
                    uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_w = (
                        torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1)
                        .min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_w
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.solver.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    delete_additional_ckpt(save_dir, 1)
                    accelerator.save_state(save_path)

            progress_bar.set_postfix(
                step_loss=loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0],
            )

            if global_step >= cfg.solver.max_train_steps:
                break

        if (epoch + 1) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(unwrap_net.reference_unet, save_dir, "reference_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.denoising_unet, save_dir, "denoising_unet", global_step, total_limit=3)
            save_checkpoint(unwrap_net.pose_guider, save_dir, "pose_guider", global_step, total_limit=3)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/stage1.yaml")
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        cfg = OmegaConf.load(args.config)
    elif args.config.endswith(".py"):
        cfg = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")

    main(cfg)
