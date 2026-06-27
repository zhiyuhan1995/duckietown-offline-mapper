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
