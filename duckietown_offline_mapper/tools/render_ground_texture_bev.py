from __future__ import annotations

import argparse
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.ground_texture import render_ground_texture_bev  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a VGGT camera-guided ground-plane BEV texture.")
    parser.add_argument("--run-summary", required=True, help="Path to a mapper run_summary.yaml.")
    parser.add_argument("--output-dir", required=True, help="Directory for BEV texture outputs.")
    parser.add_argument("--resolution", type=float, default=0.005, help="BEV meters per pixel.")
    parser.add_argument("--fusion-mode", choices=["best_view", "weighted_mean"], default="best_view")
    parser.add_argument("--padding", type=float, default=0.0, help="Extra meters around summary BEV bounds.")
    parser.add_argument("--confidence-scale", type=float, default=None)
    parser.add_argument("--min-weight", type=float, default=1e-5)
    parser.add_argument("--view-angle-power", type=float, default=1.5)
    parser.add_argument("--distance-power", type=float, default=1.0)
    parser.add_argument("--border-margin-px", type=float, default=4.0)
    parser.add_argument("--inpaint-radius", type=int, default=3)
    parser.add_argument("--unknown-rgb", type=int, nargs=3, default=(80, 80, 80))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_ground_texture_bev(
        run_summary_path=args.run_summary,
        output_dir=args.output_dir,
        resolution=args.resolution,
        fusion_mode=args.fusion_mode,
        padding=args.padding,
        confidence_scale=args.confidence_scale,
        min_weight=args.min_weight,
        view_angle_power=args.view_angle_power,
        distance_power=args.distance_power,
        border_margin_px=args.border_margin_px,
        inpaint_radius=args.inpaint_radius,
        unknown_rgb=tuple(int(v) for v in args.unknown_rgb),
    )
    print(f"Exported ground texture BEV to {Path(args.output_dir)}")
    print(result.stats)


if __name__ == "__main__":
    main()
