from __future__ import annotations

import argparse
from pathlib import Path

from src.io_utils import deep_update, load_yaml, save_yaml
from src.pipeline import load_default_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Duckietown offline semantic-occupancy BEV mapper.")
    parser.add_argument("--config", default=None, help="Optional YAML config override.")
    parser.add_argument("--input", default=None, help="Video path or image folder.")
    parser.add_argument("--output", default=None, help="Output directory.")
    parser.add_argument("--keyframe-interval", type=int, default=None)
    parser.add_argument("--max-keyframes", type=int, default=None)
    parser.add_argument("--resolution", type=float, default=None)
    args = parser.parse_args()

    config = load_default_config()
    if args.config:
        config = deep_update(config, load_yaml(args.config))
    if args.input:
        config["input"]["path"] = args.input
    if args.output:
        config["export"]["output_dir"] = args.output
    if args.keyframe_interval is not None:
        config["input"]["keyframe_interval"] = args.keyframe_interval
    if args.max_keyframes is not None:
        config["input"]["max_keyframes"] = args.max_keyframes
    if args.resolution is not None:
        config["bev"]["resolution"] = args.resolution

    result = run_pipeline(config)
    out = Path(config["export"]["output_dir"])
    save_yaml({"config": config, "result": result}, out / "run_summary.yaml")
    print(f"Exported map to {out}")
    print(result["stats"])


if __name__ == "__main__":
    main()

