import argparse
import json
import os

# -----
# Produces:
# [
#   {"video_path": "...", "kps_path": "..."},
#   ...
# ]
# -----
# Example:
# python tools/extract_meta_info.py \
#   --video_root /path/to/videos \
#   --dwpose_root /path/to/dwpose_output \
#   --dataset_name fashion \
#   --mapping_mode basename
# -----

def collect_mp4s(video_root: str):
    video_mp4_paths = []
    for root, _, files in os.walk(video_root):
        for name in files:
            if name.lower().endswith(".mp4"):
                video_mp4_paths.append(os.path.join(root, name))
    # stable order
    video_mp4_paths.sort()
    return video_mp4_paths

def build_kps_path(video_path: str, video_root: str, dwpose_root: str, mapping_mode: str):
    if mapping_mode == "basename":
        # dwpose_root/<video_basename>
        return os.path.join(dwpose_root, os.path.basename(video_path))

    if mapping_mode == "relative":
        # dwpose_root/<relative_dir>/<video_basename>
        rel = os.path.relpath(video_path, video_root)  # e.g. subset/a/b/c.mp4
        rel_dir = os.path.dirname(rel)                 # e.g. subset/a/b
        return os.path.join(dwpose_root, rel_dir, os.path.basename(video_path))

    raise ValueError(f"Unknown mapping_mode: {mapping_mode}. Use 'basename' or 'relative'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True, help="Root folder containing .mp4 videos")
    parser.add_argument("--dwpose_root", type=str, required=True, help="Root folder containing DWpose outputs")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--meta_info_name", type=str, default=None)
    parser.add_argument(
        "--mapping_mode",
        type=str,
        default="basename",
        choices=["basename", "relative"],
        help="How to map videos to DWpose outputs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data",
        help="Output directory for meta json",
    )

    args = parser.parse_args()

    meta_name = args.meta_info_name or args.dataset_name
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{meta_name}_meta.json")

    video_mp4_paths = collect_mp4s(args.video_root)

    meta_infos = []
    missing = 0

    for video_path in video_mp4_paths:
        kps_path = build_kps_path(video_path, args.video_root, args.dwpose_root, args.mapping_mode)
        if os.path.exists(kps_path):
            meta_infos.append({"video_path": video_path, "kps_path": kps_path})
        else:
            missing += 1
            print(f"Warning: Keypoint file not found for {video_path}")
            print(f"  expected at: {kps_path}")

    with open(out_path, "w") as f:
        json.dump(meta_infos, f, indent=2)

    print(f"Meta info saved to {out_path}")
    print(f"Total videos found: {len(video_mp4_paths)}")
    print(f"Total videos with kps: {len(meta_infos)}")
    print(f"Missing kps: {missing}")

if __name__ == "__main__":
    main()
