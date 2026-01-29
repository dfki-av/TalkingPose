import argparse
import json
from collections import deque
from pathlib import Path

import cv2


def _to_bgr(frame):
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def temporal_jitter_avg(real_path: str, gen_path: str, delta: int):
    cap_r = cv2.VideoCapture(real_path)
    cap_g = cv2.VideoCapture(gen_path)
    if not cap_r.isOpened() or not cap_g.isOpened():
        cap_r.release()
        cap_g.release()
        return None, 0, 0.0

    rbuf, gbuf = deque(maxlen=delta + 1), deque(maxlen=delta + 1)
    target_size = None
    s, n = 0.0, 0

    while True:
        rr, fr = cap_r.read()
        rg, fg = cap_g.read()
        if not (rr and rg):
            break

        fr = _to_bgr(fr)
        fg = _to_bgr(fg)

        if target_size is None:
            target_size = (fr.shape[1], fr.shape[0])

        if (fr.shape[1], fr.shape[0]) != target_size:
            fr = cv2.resize(fr, target_size, interpolation=cv2.INTER_AREA)
        if (fg.shape[1], fg.shape[0]) != target_size:
            fg = cv2.resize(fg, target_size, interpolation=cv2.INTER_AREA)

        rbuf.append(fr)
        gbuf.append(fg)

        if len(rbuf) == delta + 1:
            dr = cv2.absdiff(rbuf[-1], rbuf[0])
            dg = cv2.absdiff(gbuf[-1], gbuf[0])
            d = cv2.absdiff(dr, dg)
            m = cv2.mean(d)
            s += (m[0] + m[1] + m[2]) / 3.0
            n += 1

    cap_r.release()
    cap_g.release()

    return (s / n if n else None), n, s


def index_mp4(root: Path):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*.mp4")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True)
    ap.add_argument("--gen_dir", type=str, required=True)
    ap.add_argument("--delta", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.delta < 1:
        raise ValueError("delta must be >= 1")

    real_dir = Path(args.real_dir)
    gen_dir = Path(args.gen_dir)

    real_map = index_mp4(real_dir)
    gen_map = index_mp4(gen_dir)

    common = sorted(set(real_map) & set(gen_map))
    missing_in_real = sorted(set(gen_map) - set(real_map))
    missing_in_gen = sorted(set(real_map) - set(gen_map))

    pairs = []
    per_video_avgs = []
    total_s, total_n = 0.0, 0

    for k in common:
        avg, n, s = temporal_jitter_avg(str(real_map[k]), str(gen_map[k]), args.delta)
        pairs.append(
            {
                "key": k,
                "real": str(real_map[k]),
                "generated": str(gen_map[k]),
                "avg_tje": avg,
                "n_terms": n,
            }
        )
        if avg is not None:
            per_video_avgs.append(avg)
        total_s += s
        total_n += n

    out = {
        "delta": args.delta,
        "num_pairs": len(common),
        "overall_avg_tje": (sum(per_video_avgs) / len(per_video_avgs) if per_video_avgs else None),
        "overall_weighted_avg_tje": (total_s / total_n if total_n else None),
        "missing_in_real": missing_in_real,
        "missing_in_generated": missing_in_gen,
        "pairs": pairs,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
