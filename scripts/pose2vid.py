import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torchvision.io as tv_io
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    return parser.parse_args()


def _weight_dtype(dtype_name: str):
    return torch.float16 if dtype_name == "fp16" else torch.float32


def _load_state(model, ckpt_path, strict=True):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=strict)
    return model


def _build_pipeline(config, infer_config, dtype):
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to("cuda", dtype=dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to("cuda", dtype=dtype)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to("cuda", dtype=dtype)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to("cuda", dtype=dtype)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to("cuda", dtype=dtype)

    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))

    _load_state(denoising_unet, config.denoising_unet_path, strict=False)
    _load_state(reference_unet, config.reference_unet_path, strict=True)
    _load_state(pose_guider, config.pose_guider_path, strict=True)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    ).to("cuda", dtype=dtype)

    return pipe


def _list_unprocessed(src_dir: str, save_dir: Path):
    items = sorted(os.listdir(src_dir))
    return [name for name in items if not (save_dir / name).exists()]


def _frames_to_video_tensor(frames_chw):
    return torch.stack(
        [
            (f.permute(1, 2, 0) * 255).to(torch.uint8).cpu().contiguous()
            for f in frames_chw
        ],
        dim=0,
    )


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    dtype = _weight_dtype(config.weight_dtype)
    infer_config = OmegaConf.load(config.inference_config)

    pipe = _build_pipeline(config, infer_config, dtype)
    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    save_dir = Path(config.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    pose_vids = _list_unprocessed(config.pose_path, save_dir)
    ref_imgs = _list_unprocessed(config.reference_path, save_dir)

    prev_latents = None
    initial_noise = None

    for idx, (ref_name, pose_name) in enumerate(zip(ref_imgs, pose_vids), start=1):
        ref_path = os.path.join(config.reference_path, ref_name)
        pose_path = os.path.join(config.pose_path, pose_name)

        ref_image = read_frames(ref_path)[0]
        pose_frames = read_frames(pose_path)
        src_fps = get_fps(pose_path)

        print(f"Out of {len(pose_vids)} videos, only {idx} is processed")
        print(f"pose video has {len(pose_frames)} frames, with {src_fps} fps")

        out_frames_chw = []
        for frame_ind, pose_frame_pil in enumerate(pose_frames):
            video_frames, updated_latents, init_noise_ret = pipe(
                ref_image,
                [pose_frame_pil],
                width,
                height,
                args.L,
                args.steps,
                args.cfg,
                frame_ind,
                config.clc_gain,
                prev_latents,
                initial_noise,
                generator=generator,
            )

            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            prev_latents = updated_latents
            if frame_ind < 1:
                initial_noise = init_noise_ret

            out_frames_chw.append(video_frames[0, :, 0, :, :])

        video_tensor = _frames_to_video_tensor(out_frames_chw)

        tv_io.write_video(
            filename=os.path.join(config.output_dir, pose_name),
            video_array=video_tensor,
            fps=30,
            options={"crf": "23"},
        )


if __name__ == "__main__":
    main()
