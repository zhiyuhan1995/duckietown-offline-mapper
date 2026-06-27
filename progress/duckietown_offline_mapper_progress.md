# Duckietown Offline Mapper Progress

Date: 2026-06-26

## Goal Alignment

- `goal.md` is the source of truth.
- `story.md` is empty, so no additional story constraints were available.
- Input video: `track.mp4`, 1280x720, 30 FPS, 1523 frames, 50.766 seconds.

## Current Implementation

- Python project skeleton under `duckietown_offline_mapper/`.
- Modular source files for IO, keyframes, reconstruction, point clouds, plane fitting, alignment, BEV rasterization, segmentation, occupancy fusion, export, and pipeline orchestration.
- Only supported reconstruction backend: `vggt_sfm`.
- VGGT-SfM recovers cameras, intrinsics, depth-derived global point clouds, confidence, optional COLMAP sparse export, and optional bundle adjustment.
- Ground-plane alignment maps the dominant plane to `z=0` before BEV rasterization.
- Sim(3) Umeyama alignment and planar Sim(2)/SE(2) control-point alignment.
- World/grid conversion utilities.
- Threshold-based BEV semantic segmentation.
- Non-ground obstacle projection and robot-radius inflation.
- ROS-compatible `map.png`, `map.yaml`, and `occupancy_grid.npy` export.
- Streamlit UI with requested workflow tabs.
- Unit tests for alignment, plane alignment, backend selection, grid conversion, and occupancy fusion.

## Historical IPM Prototype

- An earlier IPM/dummy prototype existed only to exercise the downstream mapper before VGGT dependencies were installed.
- That prototype is superseded and removed from supported backend selection.

## Exported Artifacts

- `outputs/track_map/aligned_point_cloud.ply`
- `outputs/track_map/bev_rgb.png`
- `outputs/track_map/semantic_mask.png`
- `outputs/track_map/obstacle_occupancy.png`
- `outputs/track_map/final_occupancy_grid.png`
- `outputs/track_map/semantic_grid.npy`
- `outputs/track_map/obstacle_grid.npy`
- `outputs/track_map/occupancy_grid.npy`
- `outputs/track_map/map.png`
- `outputs/track_map/map.yaml`
- `outputs/track_map/map_metadata.yaml`
- `outputs/track_map/run_summary.yaml`

## Current Limitations

- Superseded on 2026-06-26: IPM/dummy fallback backends were removed at the user's request.
- The only supported reconstruction backend is now `vggt_sfm`.
- Full VGGT-SfM verification was run on `pulsatilla`, not on `cgpool1904`.

## VGGT-SfM Completion Update

Date: 2026-06-26

### Environment

- Host: `pulsatilla`
- GPUs: 3x NVIDIA GeForce RTX 4090
- Environment: `/home/hanzhiyu/projects/duckietown/.conda-vggt`
- Python: 3.10
- Key installed packages:
  - `torch==2.3.1+cu121`
  - `torchvision==0.18.1+cu121`
  - `vggt` from official `facebookresearch/vggt` commit `a288dd0f14786c93483e45524328726ab7b1b4ce`
  - `open3d==0.19.0`
  - `pycolmap==3.10.0`
  - `pyceres==2.3`
  - `lightglue` from `jytime/LightGlue`

### Code Changes

- Removed IPM/dummy reconstruction alternatives from the supported backend selection.
- Added `VGGT_SfMReconstructionBackend`.
- VGGT-SfM now:
  - extracts uniformly sampled keyframes from `track.mp4`
  - runs VGGT on GPU
  - predicts cameras, intrinsics, depth, and confidence
  - unprojects depth to world points
  - fuses confident colored points into `vggt_point_cloud.ply`
  - optionally exports COLMAP sparse files
  - optionally runs VGGT-SfM tracker + pycolmap bundle adjustment
- Added ground-plane alignment before BEV generation.
- Added auto ROI from reconstructed point-cloud percentiles.
- Updated Streamlit UI to expose only VGGT-SfM controls.
- Updated README and requirements for pulsatilla GPU execution.

### Verification

- Dependency/CUDA import check on pulsatilla:
  - Torch CUDA available: true
  - CUDA device count: 3
  - GPU 0: NVIDIA GeForce RTX 4090
- Unit tests:
  - Command: `.conda-vggt/bin/python -m pytest duckietown_offline_mapper/tests -q`
  - Result: `8 passed`
- Compile check:
  - Command: `.conda-vggt/bin/python -m compileall -q duckietown_offline_mapper`
  - Result: passed

### VGGT-SfM Runs

- Smoke run, 3 keyframes:
  - Output: `outputs/track_map_vggt_smoke`
  - Raw points: 10002
  - Cropped points: 9996
  - Occupancy: free 348, occupied 242, unknown 3766
- COLMAP export smoke, 3 keyframes:
  - Output: `outputs/track_map_vggt_colmap_smoke`
  - COLMAP files: `cameras.bin`, `images.bin`, `points3D.bin`
  - Raw points: 10002
  - Cropped points: 9996
  - Occupancy: free 349, occupied 247, unknown 3760
- Bundle adjustment smoke, 2 keyframes:
  - Output: `outputs/track_map_vggt_ba_smoke`
  - pycolmap BA final cost: 0.202882 px
  - Valid track count: 1184
  - Raw points: 6587
  - Cropped points: 6582
  - Occupancy: free 360, occupied 290, unknown 2094
- Final default VGGT-SfM map, 12 keyframes:
  - Output: `outputs/track_map`
  - Backend recorded in metadata: `vggt_sfm`
  - COLMAP direct sparse export: `outputs/track_map/work/colmap_sparse`
  - Raw points: 36838
  - Cropped points: 36838
  - Occupancy: free 430, occupied 212, unknown 2548
  - ROS map files: `map.png`, `map.yaml`, `occupancy_grid.npy`

### UI

- Streamlit is running on pulsatilla:
  - Network URL: `http://134.2.169.172:8501`
  - Local pulsatilla smoke check: HTTP `200`

## Local Machine Run

Date: 2026-06-26 22:04 CEST

- Host: `cgpool1904`
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB
- Environment: `/home/hanzhiyu/projects/duckietown/.conda-vggt`
- CUDA check:
  - Torch CUDA available: true
  - CUDA device count: 1
  - GPU 0: NVIDIA GeForce RTX 3090
- Unit tests:
  - Command: `.conda-vggt/bin/python -m pytest duckietown_offline_mapper/tests -q`
  - Result: `8 passed`
- Compile check:
  - Command: `.conda-vggt/bin/python -m compileall -q duckietown_offline_mapper`
  - Result: passed
- Local VGGT-SfM run:
  - Command: `CUDA_VISIBLE_DEVICES=0 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py --output outputs/track_map_local`
  - Output: `outputs/track_map_local`
  - Backend recorded in metadata: `vggt_sfm`
  - Keyframes: 12
  - Raw points: 36833
  - Cropped points: 36833
  - Occupancy: free 434, occupied 195, unknown 2561
  - COLMAP direct sparse export: `outputs/track_map_local/work/colmap_sparse`
  - ROS map files: `map.png`, `map.yaml`, `occupancy_grid.npy`
- Local Streamlit UI:
  - Network URL: `http://134.2.169.194:8501`
  - HTTP smoke check: status `200`

## Point Cloud Viewer Update

Date: 2026-06-26

- Added an interactive Plotly WebGL point-cloud viewer to the Streamlit `Reconstruction` tab.
- Default PLY source: `outputs/track_map_local/aligned_point_cloud.ply` when present.
- Viewer supports:
  - mouse drag rotation
  - scroll zoom
  - RGB / height / confidence / solid color modes
  - display point downsampling control
  - point-size control
  - latest in-memory reconstruction or exported PLY source
- Local verification:
  - Loaded `outputs/track_map_local/aligned_point_cloud.ply`
  - Point count: 36833
  - Unit tests: `8 passed`
  - Compile check: passed
  - Streamlit HTTP check: status `200`

## Black Road Point Retention Update

Date: 2026-06-26

- Diagnosis: black Duckietown road pixels are low-texture and tend to receive low VGGT confidence; `confidence_threshold: 5.0` removed most of them.
- Added `relax_ground_confidence` and `ground_confidence_threshold` to retain black road / white line / yellow line / red stop-line colored pixels at a lower confidence threshold.
- Default settings:
  - `use_point_map: false`
  - `confidence_threshold: 5.0`
  - `relax_ground_confidence: true`
  - `ground_confidence_threshold: 1.2`
- Regenerated `outputs/track_map` on pulsatilla only.
- New pulsatilla result:
  - Raw points: 145703
  - Cropped points: 145428
  - Black-ish points in `work/vggt_point_cloud.ply`: 37289
  - White-ish points: 75178
  - Yellow-ish points: 8690
  - Occupancy: free 1239, occupied 563, unknown 2167
- Access setup:
  - Streamlit runs on pulsatilla `127.0.0.1:8501`
  - Local SSH tunnel maps it to local browser URL `http://localhost:8501`

## Road Edge Completion Run

Date: 2026-06-26

- User observed road-edge gaps after black-road confidence relaxation.
- Added `duckietown_offline_mapper/configs/pulsatilla_edge_complete.yaml`.
- Edge-complete settings:
  - keyframe interval: 60
  - max keyframes: 24
  - `use_point_map: false`
  - `confidence_threshold: 4.0`
  - `ground_confidence_threshold: 1.0`
  - `sample_stride: 1`
  - `max_points: 900000`
  - ROI percentiles: 0.25 to 99.75
  - COLMAP export disabled for this dense visualization run
- Ran on pulsatilla GPU 1:
  - Command: `CUDA_VISIBLE_DEVICES=1 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py --config duckietown_offline_mapper/configs/pulsatilla_edge_complete.yaml`
  - Output: `outputs/track_map_edge_complete`
  - Raw points: 900000
  - Cropped points: 899590
  - Occupancy: free 3429, occupied 433, unknown 2778
- Point-color comparison:
  - Previous `outputs/track_map/work/vggt_point_cloud.ply`: black 35510, white 164104, yellow 4975
  - Edge-complete `outputs/track_map_edge_complete/work/vggt_point_cloud.ply`: black 296603, white 418922, yellow 38714
- UI path to inspect:
  - `outputs/track_map_edge_complete/aligned_point_cloud.ply`

## Cluster GPU01 Blackwell Run

Date: 2026-06-26

- Host: `cluster-gpu01`
- User-selected device: physical GPU 1
- Runtime binding: `CUDA_VISIBLE_DEVICES=1`
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB
- Environment: `/home/hanzhiyu/projects/duckietown/.conda-vggt-bw`
- Reason for new environment:
  - Existing `.conda-vggt` uses torch `2.3.1+cu121`, which cannot run CUDA kernels on Blackwell `sm_120`.
  - New environment uses torch `2.12.1+cu129`.
- CUDA verification:
  - Visible CUDA device count: 1
  - Visible GPU 0 maps to physical GPU 1
  - Matrix multiplication smoke test: passed
- Import verification:
  - `numpy 1.26.4`
  - `open3d 0.19.0`
  - `streamlit 1.58.0`
  - `plotly 6.8.0`
  - `vggt`, `pycolmap`, `pyceres`, `lightglue`: import passed
- Unit tests:
  - Command: `ssh cluster-gpu01 'cd /home/hanzhiyu/projects/duckietown; .conda-vggt-bw/bin/python -m pytest duckietown_offline_mapper/tests -q'`
  - Result: `8 passed`
- VGGT-SfM smoke run:
  - Command: `CUDA_VISIBLE_DEVICES=1 .conda-vggt-bw/bin/python duckietown_offline_mapper/run_pipeline.py --config duckietown_offline_mapper/configs/pulsatilla_edge_complete.yaml --output outputs/track_map_cluster_gpu01_smoke --keyframe-interval 240 --max-keyframes 3 --resolution 0.03`
  - Output: `outputs/track_map_cluster_gpu01_smoke`
  - Raw points: 394704
  - Cropped points: 394704
  - Occupancy: free 1291, occupied 57, unknown 1712
- Full edge-complete run:
  - Command: `CUDA_VISIBLE_DEVICES=1 .conda-vggt-bw/bin/python duckietown_offline_mapper/run_pipeline.py --config duckietown_offline_mapper/configs/pulsatilla_edge_complete.yaml --output outputs/track_map_cluster_gpu01_edge_complete`
  - Output: `outputs/track_map_cluster_gpu01_edge_complete`
  - Raw points: 900000
  - Cropped points: 899592
  - Occupancy: free 3433, occupied 398, unknown 2805
  - GPU memory returned to idle after run.
- UI update:
  - Point-cloud viewer default path now prefers `outputs/track_map_cluster_gpu01_edge_complete/aligned_point_cloud.ply`.
  - Alignment clickable BEV default path now prefers `outputs/track_map_cluster_gpu01_edge_complete/ground_aligned_point_cloud.ply`.
- Streamlit access:
  - Remote command binds Streamlit to `cluster-gpu01` localhost only: `127.0.0.1:8501`
  - Remote Streamlit PID: `1699721`
  - Local tunnel: `localhost:8501 -> cluster-gpu01:127.0.0.1:8501`
  - Remote HTTP check: `200`
  - Local HTTP check: `200`
  - Browser URL from this machine: `http://localhost:8501`

## Streamlit GPU Isolation Fix

Date: 2026-06-26

- Diagnosis:
  - The observed failure was not GPU OOM. `cluster-gpu01` GPU 1 only showed about 736 MiB and 0% utilization.
  - The old Streamlit service process could touch PyTorch/CUDA and keep a small CUDA context alive while serving the UI.
  - Repeated SSH restarts/probes made `cluster-gpu01` intermittently refuse or time out new SSH connections, which looked like the server had crashed.
- Fixes:
  - Removed top-level `run_pipeline` / reconstruction imports from `app.py`.
  - Removed Streamlit-side `torch.cuda.empty_cache()` calls; the UI process no longer imports torch just to clear memory.
  - Changed `Run reconstruction`, `Run BEV rasterization preview`, and `Run full export` to execute `run_pipeline.py` in an isolated subprocess.
  - The Streamlit service is now started with `CUDA_VISIBLE_DEVICES=` so the web UI process cannot see or occupy GPUs.
  - Pipeline subprocesses use `DUCKIETOWN_MAPPER_CUDA_VISIBLE_DEVICES=1`, so heavy VGGT work still runs on physical GPU 1.
  - The point-cloud `Displayed points` widget now allows up to 2,000,000 requested points and clamps to the actual PLY point count before plotting.
- Current service:
  - Remote Streamlit PID: `7803`
  - Remote HTTP check: `200`
  - Local HTTP check: `200`
  - GPU state after service startup: GPU 0 = 4 MiB, GPU 1 = 4 MiB
  - Browser URL: `http://localhost:8501`

## Alignment BEV Preview Scaling Fix

Date: 2026-06-27

- Problem:
  - The clickable alignment BEV preview was displayed at raw raster size.
  - With the default `0.01-0.015 m/cell` preview resolution and a small Duckietown map footprint, the image appeared as a tiny thumbnail.
- Fix:
  - Added `BEV display width` control to the Alignment tab.
  - Default display width is now `1200 px`.
  - The raster image is enlarged with nearest-neighbor scaling for clear point inspection.
  - Click coordinates on the enlarged image are mapped back to the original BEV raster before converting to source `x/y`; selected source `z` remains `0.0`.
- Verification:
  - `python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `python -m pytest duckietown_offline_mapper/tests -q`: `8 passed`
  - Restarted Streamlit on `cluster-gpu01` with GPU-hidden UI process.
  - Remote Streamlit PID: `16949`
  - Browser URL: `http://localhost:8501`

## Alignment BEV HD Render Fix

Date: 2026-06-27

- Problem:
  - The enlarged Alignment BEV image was still pixelated because it was a nearest-neighbor upscale of a low-resolution raster.
- Fix:
  - Replaced the upscale path with direct high-resolution point rendering.
  - The Alignment preview now projects source point-cloud coordinates directly into the requested display canvas.
  - Default `BEV display width`: `1600 px`; maximum: `2400 px`.
  - Default `BEV point radius`: `2 px`.
  - Click coordinates are converted directly from HD display pixel coordinates back to source world `x/y`; selected source `z` remains `0.0`.
- Verification:
  - `python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `python -m pytest duckietown_offline_mapper/tests -q`: `8 passed`
  - Restarted Streamlit on `cluster-gpu01` with GPU-hidden UI process.
  - Remote Streamlit PID: `20146`
  - Browser URL: `http://localhost:8501`

## Point Cloud Viewer Large-Point Render Fix

Date: 2026-06-27

- Problem:
  - Plotly `Scatter3d` could show only axes and no points when displaying more than about 250k RGB-colored points.
  - The issue is browser-side WebGL / Plotly trace-buffer pressure, not missing point-cloud data.
- Fix:
  - Split large point clouds into multiple WebGL traces.
  - RGB mode chunk size: `80,000` points per trace.
  - Height / Confidence / Solid chunk size: `120,000` points per trace.
  - Disabled hover above `200,000` points to reduce browser-side payload.
  - Viewer status now reports actual plotted points, trace count, and chunk size.
- Verification:
  - `python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `python -m pytest duckietown_offline_mapper/tests -q`: `8 passed`
  - Restarted Streamlit on `cluster-gpu01` with GPU-hidden UI process.
  - Remote Streamlit PID: `38874`
  - Browser URL: `http://localhost:8501`

## 3DGS Pipeline Upgrade Start

Date: 2026-06-27

- User requested a Level-2 3D Gaussian Splatting stage after VGGT-SfM.
- Intended updated flow:
  - VGGT-SfM reconstructs cameras, geometry, and a bootstrap point cloud.
  - A coarse RANSAC ground plane defines the initial BEV/top-down coordinate system.
  - VGGT predictions are exported as a COLMAP-style scene.
  - 3D Gaussian Splatting is trained from the VGGT-COLMAP scene.
  - A top-down orthographic BEV texture / alpha / height product is rendered from the optimized Gaussian scene.
  - Downstream crop, semantic segmentation, and occupancy should prefer the Gaussian-rendered BEV products over the raw point-cloud raster.
- Repository archive before 3DGS work:
  - Added `.gitignore` excluding conda environments, caches, videos, outputs, logs, point clouds, images, and numpy result arrays.
  - Initialized local Git repo on branch `main`.
  - Initial archive commit: `5281fa1 archive current duckietown mapper`.
  - Tracked file count: 30.
  - Confirmed ignored:
    - `track.mp4`
    - `outputs/`
    - `logs/`
    - `.conda-vggt/`
    - `.conda-vggt-bw/`
  - Created remote bare repository on `cluster-gpu03`: `/home/hanzhiyu/git/duckietown-offline-mapper.git`.
  - Added remote `origin`: `cluster-gpu03:/home/hanzhiyu/git/duckietown-offline-mapper.git`.
  - Pushed `main` to `origin/main`.
- GitHub note:
  - GitHub SSH authentication works for user `zhiyuhan1995`.
  - `gh` is not installed and no `GITHUB_TOKEN` / `GH_TOKEN` / `GITHUB_PAT` is available.
  - The available GitHub connector can operate on existing repositories but does not expose repository creation.
  - The cluster bare repo is the current remote archive; a GitHub repo can be added later when an empty repo exists or `gh` / token auth is available.
- cluster-gpu03 check:
  - Host reachable as `cluster-gpu03`.
  - GPUs: two NVIDIA GeForce RTX 4090 cards.
  - `.conda-vggt` uses torch `2.3.1+cu121` and sees both GPUs.
  - `vggt`, `pycolmap`, `cv2`, and `open3d` import successfully.
  - `gsplat` is not installed yet.
- Documentation updated:
  - Rewrote `goal.md` for the two-stage VGGT bootstrap plus Level-2 3DGS BEV pipeline.
  - Filled `story.md` with the motivation for replacing point-cloud BEV with Gaussian-rendered continuous BEV texture.

## GitHub Archive Update

Date: 2026-06-27

- User created GitHub repository: `zhiyuhan1995/duckietown-offline-mapper`.
- Added GitHub remote:
  - `github`: `git@github.com:zhiyuhan1995/duckietown-offline-mapper.git`
- Pushed archived `main` branch to GitHub.
- Latest archived commit after local environment ignore cleanup:
  - `eb2402d ignore local virtual environments`
- Mirrored the same `main` branch to the cluster bare remote:
  - `origin`: `cluster-gpu03:/home/hanzhiyu/git/duckietown-offline-mapper.git`

## 3DGS Smoke Troubleshooting Log

Date: 2026-06-27

### Issue: gsplat CUDA kernels were unavailable

- Symptom:
  - `simple_trainer.py` parsed the VGGT-COLMAP scene and initialized 40189 Gaussians, then failed on the first rasterization call.
  - Error included: `gsplat: No CUDA toolkit found. gsplat will be disabled.`
  - Follow-up exception: `AttributeError: 'NoneType' object has no attribute 'fully_fused_projection_fwd'`
- Diagnosis route:
  - Checked `nvcc`: not available on `cluster-gpu03`.
  - Checked PyTorch CUDA:
    - `.conda-vggt` has `torch 2.3.1+cu121`
    - CUDA is available and both RTX 4090 GPUs are visible.
  - Installed `nvidia-cuda-nvcc-cu12==12.1.105` into `.venv-gsplat`, but verified it only provided `ptxas` and CUDA headers, not an actual `nvcc` executable usable by PyTorch extension builds.
  - Checked the official gsplat wheel index and found a matching prebuilt wheel for PyTorch 2.3 + CUDA 12.1.
- Fix:
  - Installed prebuilt gsplat wheel into `.venv-gsplat`:
    - `gsplat-1.4.0+pt23cu121-cp310-cp310-linux_x86_64.whl`
  - Left `.conda-vggt` intact for VGGT/pycolmap compatibility.
  - Kept `.venv-gsplat` as the separate 3DGS environment, with `pycolmap.SceneManager` available for the gsplat examples.
- Verification:
  - `.venv-gsplat` imports `gsplat 1.4.0+pt23cu121` from the venv site-packages.
  - `from gsplat.rendering import rasterization` succeeds.
  - The smoke training run progressed past the first CUDA rasterization call and trained normally.
- Residual note:
  - The VGGT official docs recommend `gsplat==1.3.0`, but the local cluster lacks a usable CUDA toolkit for compiling 1.3.0 from source. The prebuilt 1.4.0 wheel is the practical smoke-test path for this host.

### Issue: smoke run failed with `ZeroDivisionError`

- Symptom:
  - A short smoke run launched with `--steps_scaler 0.002` failed after the first few iterations.
  - Error: `ZeroDivisionError: integer division or modulo by zero`.
- Diagnosis route:
  - Read `external/gsplat-v1.3.0/examples/simple_trainer.py`.
  - `Config.adjust_steps()` multiplies `DefaultStrategy.refine_every` by `steps_scaler` and casts to `int`.
  - Default `refine_every` is `100`; `int(100 * 0.002)` becomes `0`.
  - `DefaultStrategy.step_post_backward()` later evaluates `step % self.refine_every`, causing the crash.
- Fix:
  - Stopped using tiny `steps_scaler` for smoke tests.
  - Used explicit short training length while keeping strategy intervals positive:
    - `--max_steps 120`
    - `--save_steps 120`
    - `--eval_steps 999999`
  - Pushed `eval_steps` beyond the smoke run to avoid the example script's final trajectory-render path, which is not needed for checkpoint verification.
- Verification:
  - Command completed successfully on `cluster-gpu03` GPU 0.
  - Output directory: `outputs/gsplat_smoke_result_success`
  - Saved checkpoint: `outputs/gsplat_smoke_result_success/ckpts/ckpt_119_rank0.pt`
  - Saved train stats: `outputs/gsplat_smoke_result_success/stats/train_step0119_rank0.json`
  - Final reported stats:
    - step: `119`
    - GPU memory: `0.376 GiB`
    - number of Gaussians: `40189`

### Issue: gsplat scene needed manual `image_1` aliases

- Symptom:
  - The initial gsplat Parser test only worked after manually copying keyframes to names such as `images/image_1`, `images/image_2`, and `images/image_3`.
  - VGGT-COLMAP export stored COLMAP image names as `image_1`, `image_2`, etc., while the actual saved keyframes were named `frame_0000_000000.png`, etc.
  - Repeated reconstruction runs could also leave stale images in `work/vggt_images`, making gsplat's sorted image-name mapping unsafe.
- Diagnosis route:
  - Inspected `pycolmap.Reconstruction(...).images` for `outputs/track_map_vggt_colmap_smoke/work/colmap_sparse` and `outputs/track_map/work/colmap_sparse`.
  - Confirmed the stored image names were `image_1`, `image_2`, etc.
  - Confirmed `work/vggt_images` could contain more files than the current COLMAP sparse model after repeated runs.
- Fix:
  - `VGGT_SfMReconstructionBackend._write_vggt_images()` now resets the generated image directory before saving frames.
  - COLMAP sparse image names are rewritten to the actual keyframe filenames before `reconstruction.write(...)`.
  - Reconstruction export now creates `work/gsplat_scene/images` and `work/gsplat_scene/sparse` automatically for gsplat training.
- Verification:
  - Re-ran a 3-frame VGGT-COLMAP smoke on `cluster-gpu03`.
  - COLMAP image names became:
    - `frame_0000_000000.png`
    - `frame_0001_000180.png`
    - `frame_0002_000360.png`
  - `work/gsplat_scene/images` contained exactly 3 images and `work/gsplat_scene/sparse` contained `cameras.bin`, `images.bin`, and `points3D.bin`.
  - gsplat Parser loaded the generated scene directly; no manual alias files were needed.

### Issue: Gaussian BEV initially was only a standalone preview

- Symptom:
  - The first Gaussian BEV render used checkpoint-normalized coordinates and auto bounds.
  - That was useful for visualizing the splats, but it was not aligned to the mapper's occupancy-grid frame.
  - Semantic segmentation could not safely consume that preview because its pixels did not share the mapper's `x/y/resolution` metadata.
- Diagnosis route:
  - Checked gsplat's orthographic projection implementation: `pixel = camera_xy * focal + center`.
  - Reproduced the gsplat Parser normalization transform with `Parser(scene_dir, normalize=True).transform`.
  - Derived the map-to-normalized transform as:
    - `T_map_to_norm = T_colmap_raw_to_norm @ inverse(T_raw_to_map)`
  - Verified that map x/y/forward basis vectors have equal normalized scale for this scene, so an orthographic camera can render directly into mapper grid coordinates.
- Fix:
  - Added `duckietown_offline_mapper/tools/render_gaussian_bev.py`.
  - The script supports:
    - standalone auto-bounds preview rendering
    - map-aligned rendering using `scene_dir`, `raw_to_map_transform`, BEV bounds, resolution, width, height, and z bounds
  - Added `duckietown_offline_mapper/src/gaussian.py` to run gsplat training and map-aligned BEV rendering in a separate `.venv-gsplat` subprocess.
  - Updated `pipeline.py` so `gaussian.enabled: true` runs:
    - gsplat training from `work/gsplat_scene`
    - map-aligned Gaussian BEV rendering
    - semantic segmentation using the Gaussian BEV when `gaussian.use_for_bev: true`
  - Obstacle occupancy still comes from non-ground point-cloud projection.
- Verification:
  - Standalone Gaussian preview:
    - Output: `outputs/gsplat_smoke_named_bev/gaussian_bev_rgb.png`
    - Size: `1600 x 1423`
    - Nonzero alpha pixels: `2241541`
  - Map-aligned render smoke:
    - Output: `outputs/gsplat_smoke_named_bev_map_aligned/gaussian_bev_rgb.png`
    - Size: `74 x 74`
    - Nonzero alpha pixels: `5298`
  - Full pipeline smoke:
    - Config: `duckietown_offline_mapper/configs/cluster_gpu03_gaussian_smoke.yaml`
    - Host/GPU: `cluster-gpu03`, `CUDA_VISIBLE_DEVICES=0`
    - Output: `outputs/track_map_gaussian_smoke`
    - `bev_generation.source`: `gaussian_3dgs`
    - Gaussian render mode: `map_aligned`
    - Gaussian checkpoint: `outputs/track_map_gaussian_smoke/work/gaussian_splatting/ckpts/ckpt_119_rank0.pt`
    - Gaussian BEV: `outputs/track_map_gaussian_smoke/work/gaussian_bev/gaussian_bev_rgb.png`
    - Final stats:
      - raw points: `53928`
      - cropped points: `53856`
      - free cells: `3623`
      - occupied cells: `1853`
      - unknown cells: `0`
- Residual note:
  - This is still a 3-frame / 120-step smoke test. It validates the Level-2 pipeline wiring, not final map quality.
  - A production-quality Duckietown BEV should use more keyframes, more gsplat steps, and a tuned alpha/ROI policy so Gaussian opacity does not over-mark unknown space as observed.

## 3DGS Quality Gate Retraction

### Issue: Gaussian BEV was connected before render quality was acceptable

- Symptom:
  - The user reviewed the Gaussian BEV render and correctly flagged that it was not realistic or continuous enough to feed into the mapping pipeline.
  - The rendered BEV still had smoke-test artifacts: blurred texture, incomplete road edges, irregular coverage, and poor suitability for downstream semantic segmentation.
  - The app and README still exposed controls that made the Gaussian BEV look like an accepted pipeline input.
- Diagnosis route:
  - Compared the current render quality against the intended goal: a continuous top-down Duckietown texture suitable for semantic segmentation.
  - Confirmed the prior tests were only 3-frame/120-step and 6-frame/600-step smoke experiments, not formal 3DGS training.
  - Identified that smoke renders should validate scene export and renderer plumbing only, not justify downstream integration.
  - Confirmed the formal mapper must keep using the point-cloud BEV until a standalone Gaussian result passes visual QA.
- Fix:
  - Removed Gaussian execution from `src/pipeline.py`; semantic segmentation and occupancy no longer consume Gaussian BEV products.
  - Removed the `gaussian:` block from `configs/default.yaml`.
  - Deleted the pipeline Gaussian runner `src/gaussian.py` and old Gaussian pipeline configs.
  - Kept `tools/render_gaussian_bev.py` as a standalone QA renderer only.
  - Changed the Streamlit tab to `3DGS QA` and removed controls that could enable Gaussian BEV for semantic/occupancy.
  - Renamed BEV click helpers to `src/alignment_bev.py`; alignment point selection is now explicitly based on the ground-aligned point-cloud BEV.
  - Updated `goal.md` and README so Gaussian BEV is described as a quality-gate experiment, not an accepted planner input.
- Next route:
  - Re-export a denser VGGT-COLMAP scene using relaxed Duckietown road/line confidence masks.
  - Train 3DGS standalone on `cluster-gpu03` GPU 1 for a real step budget.
  - Render high-resolution BEV QA products and inspect them visually before any downstream integration is reconsidered.

### Issue: 10k 3DGS top-down BEV is unusable despite formal training

- Symptom:
  - A 12-keyframe, 900k-point VGGT-COLMAP scene was trained with gsplat for 10k steps on `cluster-gpu03` GPU 1.
  - Training completed successfully:
    - checkpoint: `outputs/track_map_3dgs_qa_dense_scene/gaussian_training_dense_10k_v1/ckpts/ckpt_9999_rank0.pt`
    - final Gaussians: `1304928`
  - Top-down orthographic BEV render was still unusable:
    - output: `outputs/track_map_3dgs_qa_dense_scene/gaussian_bev_dense_10k_ground_w2400/gaussian_bev_rgb.png`
    - size: `2400 x 2594`
    - artifact: bright streaking / floaters / warped road texture
  - Ground-plane center projection reduced the streaking but still produced a point-cloud-like texture with wrong / unstable track layout:
    - output: `outputs/track_map_3dgs_qa_dense_scene/gaussian_plane_bev_dense_10k_z006_sigma3_w2400/gaussian_plane_bev_rgb.png`
- Diagnosis route:
  - Rendered the trained 3DGS from the original COLMAP camera views and compared against the source keyframes:
    - all-view contact sheet: `outputs/track_map_3dgs_qa_dense_scene/camera_view_renders_dense_10k/camera_view_contact_sheet.png`
    - train-only contact sheet: `outputs/track_map_3dgs_qa_dense_scene/camera_view_renders_dense_10k_train_only/camera_view_contact_sheet.png`
  - Train-only camera views matched well:
    - mean PSNR: `38.28 dB`
    - most training views: `36-41 dB`
  - Held-out views `0` and `8` were poor because gsplat's default `test_every=8` excludes them from training.
  - Conclusion:
    - The 3DGS training/camera flow is basically sane for observed camera views.
    - The failure is the extreme unobserved top-down BEV render, not a simple “not enough steps” problem.
    - Standard perspective-trained 3DGS is not sufficient by itself for planner-grade orthographic BEV texture from this monocular sequence.
- Next route:
  - Do not connect this output to the mapper pipeline.
  - If 3DGS remains the desired Level-2 route, add BEV-specific supervision / constraints instead of relying on a novel top-down render:
    - train with additional synthesized/optimized overhead supervision,
    - constrain or regularize ground-plane Gaussians,
    - or optimize a dedicated 2D BEV texture from multi-view projections after geometry alignment.

### Issue: 508 training frames are for 3DGS, not for a single VGGT pass

- Symptom:
  - The user requested every 3rd video frame for Gaussian training.
  - A first attempt to run VGGT on all every-3 frames failed with CUDA OOM, which is expected for 508 frames and was the wrong interpretation of the request.
- Diagnosis route:
  - Verified `track.mp4` metadata:
    - total frames: `1523`
    - FPS: `30.00019698093881`
    - every-3 selection: `508` frames, frame indices `0, 3, 6, ..., 1521`
  - Clarified the intended separation:
    - VGGT seed reconstruction should use a small keyframe set for global geometry / map anchoring.
    - 3DGS training should use the dense every-3 frame set, but every training image still needs a camera pose.
  - Directly using VGGT output is insufficient for 508-frame 3DGS training unless VGGT has produced poses for all 508 images.
  - The raw 1280x720 video frames and VGGT's exported sparse model also do not share guaranteed 2D feature coordinates, so blindly registering raw frames into the VGGT sparse model is fragile.
- Fix route in progress:
  - Extracted 508 raw training frames to `outputs/track_map_3dgs_qa_every3_colmap_scene/images`.
  - Built SIFT features for all 508 frames using CPU `pycolmap` because the available pycolmap wheel has no CUDA SIFT support.
  - Added `duckietown_offline_mapper/tools/prepare_every3_gaussian_scene.py` as a resumable every-N-frame COLMAP scene-prep utility.
  - Added `duckietown_offline_mapper/tools/align_colmap_scene_to_vggt.py` to align the dense 508-frame SfM scene back to the VGGT seed coordinate frame using shared frame camera centers.
  - Patched the local gsplat COLMAP dataset so `test_every <= 0` means no validation holdout; this is required for all registered every-3 frames to enter training.
- Current run:
  - Host/GPU: `cluster-gpu03`, GPU 1 reserved for later 3DGS training.
  - COLMAP feature/matching is CPU-only in the current pycolmap environment.
  - Sequential matching is running from a clean `matches` / `two_view_geometries` state after an earlier monitor query locked SQLite.

### Issue: cluster-gpu03 overload / ssh refused during 508-frame 3DGS training

- Symptom:
  - After starting 508-frame gsplat training on `cluster-gpu03` GPU 1, new SSH connections to `cluster-gpu03` returned `Connection refused`.
  - The user's GPU monitor showed GPU 0 at `23972 MiB / 24564 MiB` under another user's `python` process and GPU 1 with the mapper training process at about `1324 MiB`.
  - Training log stopped updating at about `3252 / 30000` steps and no checkpoint had been written yet.
- Diagnosis route:
  - The mapper training command explicitly used `CUDA_VISIBLE_DEVICES=1`, so it did not allocate on physical GPU 0.
  - GPU 0 pressure was from another user's process, not the mapper run.
  - The mapper run on GPU 1 was low-memory at the time of inspection, but keeping it alive while `cluster-gpu03` refused SSH was unsafe.
  - Direct SSH and side-path SSH via `cluster-gpu01` / `cluster-gpu04` to `cluster-gpu03` both failed with `Connection refused`.
- Immediate action:
  - Sent interrupt to the active SSH training session.
  - Verified locally that no `simple_trainer` process remained on the local host; remote confirmation is blocked until `cluster-gpu03` accepts SSH again.
  - If the GPU monitor still shows `PID 4130324` for `hanzhiyu .venv-gsplat/bin/python`, kill that orphan process on `cluster-gpu03`.
- Follow-up:
  - Do not restart long 3DGS training on `cluster-gpu03` until the host is reachable and GPU/process ownership is clear.
  - Prefer a resumable shorter checkpoint cadence next time so a host interruption does not lose the run before the first checkpoint.

### Issue: GPU monitor looked like the mapper crashed cluster-gpu03

- Symptom:
  - The user's monitor showed `cluster-gpu03` GPU 0 at `24517 / 24564 MiB`, `99%` utilization under user `fischer`.
  - The same monitor showed the mapper's `.venv-gsplat/bin/python` under user `hanzhiyu` on GPU 1 at about `1830 MiB`.
  - The node then became unreliable for new SSH / process queries.
- Diagnosis route:
  - Rechecked the launch command: the mapper training process was started with `CUDA_VISIBLE_DEVICES=1`.
  - Reconnected to `cluster-gpu03` when possible and confirmed the host responded, but GPU/process queries were unreliable.
  - Local process inspection showed no local `simple_trainer` job left behind.
- Conclusion:
  - The screenshot does not show the mapper filling GPU 0; GPU 0 was occupied by another user's process.
  - Running long training on the same unhealthy node was still a mistake operationally, because a node-level failure can kill or strand the mapper job even when our own VRAM use is low.
- Fix / policy:
  - Treat `cluster-gpu03` as unavailable for long 3DGS training until it is stable and process ownership is clear.
  - Use shorter checkpoint intervals for the next full 3DGS run.
  - Do not start another formal 3DGS training run until the standalone QA setup and target GPU are explicitly verified.

## 3DGS Standalone-Only Retraction Finalization

Date: 2026-06-27

- The codebase was found to still contain the earlier Gaussian pipeline wiring:
  - `src/pipeline.py` could call the Gaussian runner and replace `bev_rgb` with a Gaussian render.
  - Streamlit exposed a `Run Gaussian BEV pipeline` button.
  - `README.md` and `goal.md` still described Gaussian BEV as a semantic input when enabled.
- Final correction:
  - Removed Gaussian execution from `src/pipeline.py`.
  - Removed the default `gaussian:` config block.
  - Removed old Gaussian pipeline configs and the pipeline Gaussian subprocess runner.
  - Renamed the UI tab to `3DGS QA` and limited it to inspection of standalone render artifacts and logs.
  - Kept standalone QA tools:
    - `tools/prepare_every3_gaussian_scene.py`
    - `tools/align_colmap_scene_to_vggt.py`
    - `tools/render_gsplat_camera_views.py`
    - `tools/render_gaussian_plane_bev.py`
    - `tools/render_gaussian_bev.py`
- Current quality gate:
  - Semantic segmentation and occupancy export use the point-cloud BEV raster.
  - Gaussian renders are debug / QA artifacts only.
  - A future promotion to pipeline input requires realistic source-camera renders and a stable, planner-readable BEV texture.

## 3DGS 508-Frame Formal Training QA

Date: 2026-06-27

- Training setup:
  - Dense scene: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene`
  - Source frames: every 3rd frame from `track.mp4`, `508` images total.
  - Camera prep: dense COLMAP/SfM scene aligned back to the VGGT seed coordinate frame.
  - Host/GPU: `cluster-gpu02`, physical GPU 2, because `cluster-gpu01` has Blackwell compatibility issues with the current gsplat wheel and `cluster-gpu03` was unstable / occupied.
- Completed runs:
  - 7k checkpoint dir: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/gaussian_training_every3_7k_gpu02_v1/ckpts`
  - 30k checkpoint dir: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/gaussian_training_every3_30k_gpu02_v1/ckpts`
  - Final 30k checkpoint: `ckpt_29999_rank0.pt`
  - Final 30k Gaussian count: `745855`
  - Final 30k reported memory in trainer log: about `1.48 GiB`
- Camera-view QA:
  - 7k subset contact sheet: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/camera_view_renders_every3_7k_gpu02_subset/camera_view_contact_sheet.png`
  - 7k subset mean PSNR: `21.40 dB`
  - 30k subset contact sheet: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/camera_view_renders_every3_30k_gpu02_subset/camera_view_contact_sheet.png`
  - 30k subset mean PSNR: `23.61 dB`
  - Result: 30k improved over 7k, but camera renders are still smoothed / gray and visibly imperfect.
- BEV QA:
  - Direct orthographic Gaussian BEV: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/gaussian_bev_every3_30k_gpu02_map_w2400_sh0/gaussian_bev_rgb.png`
  - Ground-plane center-splat BEV: `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/gaussian_plane_bev_every3_30k_gpu02_z006_sigma3_w2400/gaussian_plane_bev_rgb.png`
  - Result: direct orthographic BEV is still a blurred shiny smear; plane projection is more stable but still sparse / point-like and missing continuous topology.
- Conclusion:
  - More frames plus a formal 30k standard 3DGS run did not produce a planner-grade BEV texture.
  - This result must remain standalone QA only and must not be connected back into semantic segmentation / occupancy generation.
  - Next viable route is a BEV-specific method: ground-plane-constrained Gaussian / 2D texture optimization with multi-view supervision, not plain perspective-trained 3DGS rendered from an extreme overhead view.

### Issue: interactive rendering access for the trained 30k Gaussian result

- Symptom:
  - Static exported PNGs were not enough to inspect whether the 3DGS result itself is salvageable.
  - The user requested a real-time, mouse-interactive rendering UI.
- Diagnosis route:
  - Confirmed the latest usable checkpoint exists:
    - `outputs/track_map_3dgs_qa_every3_aligned_vggt_scene/gaussian_training_every3_30k_gpu02_v1/ckpts/ckpt_29999_rank0.pt`
  - Confirmed `cluster-gpu03` GPU 0 was occupied by another user's process:
    - user: `fischerp`
    - command: `radfoam/train.py`
    - memory: about `22.7 GiB`
  - Confirmed `cluster-gpu02` GPU 2 was available and compatible with `.venv-gsplat`.
- Fix route:
  - First viewer launch used `PYTHONPATH=external/gsplat-v1.3.0:external/gsplat-v1.3.0/examples`, which accidentally shadowed the installed gsplat wheel with the source tree.
  - Symptom from that bad launch:
    - `gsplat: No CUDA toolkit found. gsplat will be disabled.`
    - render request failed with `AttributeError: 'NoneType' object has no attribute 'fully_fused_projection_packed_fwd'`
  - Corrected launch to match the successful training/render commands:
    - `PYTHONPATH=external/gsplat-v1.3.0/examples`
    - this keeps the prebuilt gsplat CUDA wheel active.
  - Restarted gsplat's `simple_viewer.py` on `cluster-gpu02` GPU 2.
  - Remote listener:
    - host: `cluster-gpu02`
    - port: `8097`
    - process: `.venv-gsplat/bin/python external/gsplat-v1.3.0/examples/simple_viewer.py`
    - current PID: `2268653`
    - checkpoint: `ckpt_29999_rank0.pt`
    - observed viewer GPU memory: about `930 MiB`
  - Added local SSH port forwarding:
    - local URL: `http://127.0.0.1:8097`
    - forward: `8097 -> cluster-gpu02:8097`
- Verification:
  - Local HTTP probe returned status `200` twice for `http://127.0.0.1:8097`.
  - Remote `viser` log reports both HTTP and websocket endpoints on port `8097`.
