from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .bev import BevMetadata


def run_gaussian_bev_stage(
    config: dict[str, Any],
    scene_dir: str | Path,
    output_dir: str | Path,
    raw_to_map_transform: np.ndarray,
    metadata: BevMetadata,
    z_bounds: tuple[float, float],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = _resolve_path(project_root, scene_dir)

    python_executable = _resolve_path(project_root, config.get("python", ".venv-gsplat/bin/python"))
    trainer_script = _resolve_path(
        project_root,
        config.get("trainer_script", "external/gsplat-v1.3.0/examples/simple_trainer.py"),
    )
    examples_dir = _resolve_path(
        project_root,
        config.get("examples_dir", "external/gsplat-v1.3.0/examples"),
    )
    render_script = _resolve_path(
        project_root,
        config.get("render_script", "duckietown_offline_mapper/tools/render_gaussian_bev.py"),
    )

    result_dir = _resolve_path(output_dir, config.get("result_dir", "gaussian_splatting"))
    bev_dir = _resolve_path(output_dir, config.get("bev_dir", "gaussian_bev"))
    _reset_dir(result_dir)
    _reset_dir(bev_dir)

    transform_path = output_dir / "raw_to_map_transform.json"
    transform_path.write_text(
        json.dumps({"matrix": np.asarray(raw_to_map_transform, dtype=float).tolist()}, indent=2),
        encoding="utf-8",
    )

    env = os.environ.copy()
    cuda_visible_devices = config.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["PYTHONPATH"] = _prepend_env_path(str(examples_dir), env.get("PYTHONPATH", ""))

    max_steps = int(config.get("max_steps", 120))
    save_steps = int(config.get("save_steps", max_steps))
    eval_steps = int(config.get("eval_steps", 999999))
    train_cmd = [
        str(python_executable),
        str(trainer_script),
        "default",
        "--data_dir",
        str(scene_dir),
        "--data_factor",
        str(int(config.get("data_factor", 1))),
        "--result_dir",
        str(result_dir),
        "--disable_viewer",
        "--max_steps",
        str(max_steps),
        "--save_steps",
        str(save_steps),
        "--eval_steps",
        str(eval_steps),
        "--tb_every",
        str(int(config.get("tb_every", 20))),
    ]
    train_cmd.extend(str(x) for x in config.get("trainer_extra_args", []))
    train_log = output_dir / "gaussian_train.log"
    _run_logged(train_cmd, train_log, cwd=project_root, env=env)

    checkpoint = _latest_checkpoint(result_dir / "ckpts")
    render_cmd = [
        str(python_executable),
        str(render_script),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(bev_dir),
        "--scene-dir",
        str(scene_dir),
        "--raw-to-map-transform",
        str(transform_path),
        "--map-bounds",
        str(metadata.x_min),
        str(metadata.x_max),
        str(metadata.y_min),
        str(metadata.y_max),
        "--map-resolution",
        str(metadata.resolution),
        "--map-z-bounds",
        str(z_bounds[0]),
        str(z_bounds[1]),
        "--width",
        str(metadata.width),
        "--height",
        str(metadata.height),
        "--device",
        str(config.get("render_device", "cuda:0")),
        "--sh-degree",
        str(int(config.get("render_sh_degree", 0))),
    ]
    background = config.get("background", [0.32, 0.32, 0.32])
    render_cmd.extend(["--background", *(str(float(x)) for x in background)])
    render_cmd.extend(str(x) for x in config.get("render_extra_args", []))
    render_log = output_dir / "gaussian_render.log"
    _run_logged(render_cmd, render_log, cwd=project_root, env=env)

    metadata_path = bev_dir / "gaussian_bev_metadata.json"
    render_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "enabled": True,
        "scene_dir": str(scene_dir),
        "result_dir": str(result_dir),
        "bev_dir": str(bev_dir),
        "checkpoint": str(checkpoint),
        "rgb_path": str(bev_dir / "gaussian_bev_rgb.png"),
        "alpha_path": str(bev_dir / "gaussian_bev_alpha.png"),
        "depth_path": str(bev_dir / "gaussian_bev_expected_depth.png"),
        "metadata_path": str(metadata_path),
        "raw_to_map_transform_path": str(transform_path),
        "train_log": str(train_log),
        "render_log": str(render_log),
        "render_metadata": render_metadata,
        "train_command": train_cmd,
        "render_command": render_cmd,
    }


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _prepend_env_path(path: str, existing: str) -> str:
    return path if not existing else path + os.pathsep + existing


def _run_logged(command: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Gaussian subprocess failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n\nLast log lines:\n{_tail(log_path)}"
        )


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    checkpoints = sorted(
        ckpt_dir.glob("ckpt_*_rank0.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        raise RuntimeError(f"No gsplat checkpoints found in {ckpt_dir}")
    return checkpoints[-1]


def _tail(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return f"{path} does not exist"
    return "\n".join(lines[-max_lines:])
