import os
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download

BASE_DIR = ""

def prepare_image_encoder():
    print(f"Preparing image encoder weights...")
    local_dir = os.path.join(BASE_DIR, "pretrained_weights")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = os.path.join(local_dir, path)
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def prepare_dwpose():
    print(f"Preparing DWPose weights...")
    local_dir = os.path.join(BASE_DIR, "pretrained_weights/DWPose")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx",
    ]:
        path = Path(hub_file)
        saved_path = os.path.join(local_dir, path)
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="yzd-v/DWPose",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def prepare_vae():
    print(f"Preparing vae weights...")
    local_dir = os.path.join(BASE_DIR, "pretrained_weights/sd-vae-ft-mse")
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.bin",
    ]:
        path = Path(hub_file)
        saved_path = os.path.join(local_dir, path)
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )

def prepare_talkingpose():
    print("Preparing TalkingPose weights...")
    local_dir = os.path.join(BASE_DIR, "pretrained_weights", "talkingpose")
    os.makedirs(local_dir, exist_ok=True)

    repo_id = "arjavanmardi/TalkingPose"
    subfolder = "weights"  # this matches your HF repo structure

    for filename in [
        "denoising_unet.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]:
        saved_path = os.path.join(local_dir, filename)
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id=repo_id,
            subfolder=subfolder,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # avoids symlink issues on some systems
        )

if __name__ == '__main__':
    prepare_image_encoder()
    prepare_dwpose()
    prepare_vae()
    prepare_talkingpose()