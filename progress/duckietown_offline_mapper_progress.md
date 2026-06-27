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
