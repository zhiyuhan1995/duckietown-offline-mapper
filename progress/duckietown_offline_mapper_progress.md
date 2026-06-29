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

## VGGT Camera-Guided Ground Texture BEV

Date: 2026-06-28

- User-approved route:
  - Stop pursuing 3D Gaussian rendering for this mapper stage.
  - Use VGGT's reconstructed point cloud and camera intrinsics / extrinsics.
  - Fit the ground plane with RANSAC, align it to `z=0`, then inverse-project BEV ground cells into all VGGT camera views.
  - Fuse sampled image pixels into a continuous ground-plane texture BEV.
- User-selected VGGT reconstruction parameters:
  - `keyframe_interval: 60`
  - `max_keyframes: 24`
  - `confidence_threshold: 1.0`
  - `sample_stride: 1`
  - `max_points: 600000`
  - `relax_ground_confidence: true`
  - `ground_confidence_threshold: 1.0`
  - `use_point_map: false`
  - `save_colmap: true`
  - `bundle_adjustment: false`
- Implementation:
  - Added `src/ground_texture.py`.
  - Added `tools/render_ground_texture_bev.py`.
  - Added a Streamlit `Ground Texture` tab.
  - Updated default config to the user-selected VGGT settings.
- Fusion method:
  - Generate a BEV grid in the map / ground-aligned plane.
  - Transform each BEV cell center back into VGGT raw reconstruction coordinates.
  - Project that ground point into each VGGT camera with saved `camera_extrinsics.npy` and `camera_intrinsics.npy`.
  - Sample the VGGT-preprocessed image with bilinear interpolation.
  - Weight samples by view angle, camera distance, image-border margin, and VGGT confidence.
  - Export fused texture, raw texture, observed mask, weight map, and observation-count map.
- Troubleshooting:
  - Initial run failed because RGB sampling returned a full-length BEV array while the caller assigned it only to the valid subset.
  - Fixed by treating RGB samples as full-length arrays.
  - Second run exposed the same issue for confidence sampling.
  - Fixed confidence sampling the same way.
- First successful output:
  - Source run: `outputs/track_map/run_summary.yaml`
  - Source images / cameras: `24`
  - Output: `outputs/track_map/ground_texture_bev`
  - Resolution: `0.005 m/pixel`
  - Size: `280 x 266`
  - Observed pixels: `71207 / 74480`
  - Observed fraction: `0.9561`
  - Mean observations per observed pixel: `8.92`
- High-resolution output:
  - Output: `outputs/track_map/ground_texture_bev_r002`
  - Resolution: `0.002 m/pixel`
  - Size: `698 x 664`
  - Observed pixels: `443484 / 463472`
  - Observed fraction: `0.9569`
  - Mean observations per observed pixel: `8.94`
  - Visual check: produces a continuous, readable top-down Duckietown ground texture with road, lane markings, stop lines, and floor context.
- Fusion comparison:
  - `best_view` preserves sharper lane boundaries in standalone visual inspection.
  - `weighted_mean` reduces some seams and is now the user-selected default for the main pipeline.
- Verification:
  - `python -m py_compile duckietown_offline_mapper/src/ground_texture.py duckietown_offline_mapper/tools/render_ground_texture_bev.py duckietown_offline_mapper/app.py`: passed
  - `python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`

## Ground Texture Pipeline Integration

Date: 2026-06-28

- Goal:
  - After VGGT-SfM reconstruction and RANSAC ground-plane extraction, use the VGGT camera-guided IPM ground texture as the BEV image source.
  - Downstream BEV alignment preview, ROI crop, and semantic segmentation should use the continuous texture instead of the old point-cloud raster image.
  - Non-ground obstacle occupancy still uses the aligned point cloud, because obstacle evidence is 3D height-based.
- Implementation:
  - Added `ground_texture` settings to the default config.
  - Set pipeline default to `ground_texture.enabled: true` and `fusion_mode: weighted_mean`.
  - In `run_pipeline`, after ground alignment and ROI metadata creation, write a lightweight `ground_texture_input_summary.yaml` and call `render_ground_texture_bev`.
  - `bev_rgb.png` is now the exported weighted-mean ground texture when ground texture is enabled.
  - `segment_bev_rgb(...)` now receives this texture image, so semantic masks are generated from the IPM texture.
  - `map_metadata.yaml` records `bev_generation.source: vggt_camera_ground_texture` and `fusion_mode: weighted_mean`.
  - The Alignment tab now defaults to a `Ground Texture` preview source with click-to-select source `x/y`; selected source `z` is fixed to `0.0`.
- Troubleshooting:
  - Verified that the previous pipeline path rasterized `cropped_cloud` directly with `rasterize_point_cloud(...)`, producing the sparse point-style BEV.
  - Switched the default path to texture generation while keeping an explicit disabled branch for comparing the old point-cloud raster if needed.
  - Checked that enlarged texture-click coordinates are mapped by display image bounds back into the same BEV metadata extents, so the display size does not change selected world coordinates.
- Smoke test on `heracleum`:
  - Command used GPU 1 with a tiny reconstruction: `CUDA_VISIBLE_DEVICES=1 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py --config duckietown_offline_mapper/configs/default.yaml --output outputs/track_map_ground_texture_smoke --keyframe-interval 240 --max-keyframes 3 --resolution 0.01`
  - Output: `outputs/track_map_ground_texture_smoke`
  - `bev_rgb.png` and `ground_texture/ground_texture_bev.png` have the same hash, proving the exported BEV image is the ground texture.
  - `map_metadata.yaml` reports `bev_generation.source: vggt_camera_ground_texture`.
  - `map_metadata.yaml` reports `bev_generation.fusion_mode: weighted_mean`.
  - Smoke observed fraction: `0.7168`; mean observations on observed cells: `2.36`.
- Verification:
  - `.conda-vggt/bin/python -m py_compile duckietown_offline_mapper/src/pipeline.py duckietown_offline_mapper/src/ground_texture.py duckietown_offline_mapper/app.py`: passed
  - `.conda-vggt/bin/python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`
- Streamlit:
  - Service restarted on `heracleum`.
  - Remote Streamlit PID: `3995669`
  - Local tunnel is active: `localhost:8501 -> heracleum:127.0.0.1:8501`
  - Local HTTP check: `200`
  - Browser URL: `http://127.0.0.1:8501`

## Alignment Ground Texture Regeneration Fix

Date: 2026-06-28

- Problem:
  - The Alignment tab's `Ground Texture` preview was reading a previously exported `ground_texture_bev.png`.
  - The default path picker chose the newest existing texture image from several output directories.
  - This made the alignment preview capable of showing a stale fixed image rather than a freshly generated IPM result from the current reconstruction summary.
- Fix:
  - Removed the fixed PNG selector from the Alignment tab.
  - Alignment now takes a `Run summary for IPM` path and calls `render_ground_texture_bev(...)` for that summary.
  - The displayed image comes directly from the returned texture array, not from `_load_rgb_image(...)`.
  - The generated files are written under `<run_output>/alignment_ground_texture/` as reproducible artifacts.
  - A signature made from the run-summary path, run-summary modification time, IPM resolution, fusion mode, and weighting parameters controls regeneration.
  - If the signature changes, or the user presses `Regenerate alignment IPM texture`, the Alignment preview recomputes IPM.
  - If the signature is unchanged, the current in-session generated texture is reused so clicking source points does not trigger unnecessary IPM recomputation on every Streamlit rerun.
- Verification:
  - `.conda-vggt/bin/python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `.conda-vggt/bin/python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`
  - Direct heracleum IPM check wrote `outputs/track_map_ground_texture_smoke/alignment_ground_texture_test/ground_texture_bev.png` and metadata successfully.
  - Streamlit restarted on `heracleum`; remote PID: `303087`.
  - Local HTTP check: `200`.
  - Browser URL: `http://127.0.0.1:8501`

## Semantic Preview Live Update Fix

Date: 2026-06-28

- Problem:
  - Moving the Semantic tab sliders did not visibly change the preview image.
  - Root cause: the tab displayed `last_run["paths"]["semantic_mask"]`, a static PNG exported by the previous pipeline run.
  - The slider values were written to the in-memory config, but the page did not call `segment_bev_rgb(...)` again for preview.
- Fix:
  - Added a live `BEV image for live semantic preview` source path.
  - The Semantic tab now loads the current `bev_rgb.png`, applies the current slider values, calls `segment_bev_rgb(...)`, and colorizes the result in-page.
  - Added class pixel counts beside the preview so parameter changes are numerically visible even when the visual change is subtle.
  - The image loader is cached with file modification time in the key, so overwriting `bev_rgb.png` refreshes the preview instead of reusing stale cache.
- Verification:
  - `.conda-vggt/bin/python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `.conda-vggt/bin/python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`
  - Direct comparison with loose vs strict semantic thresholds produced different class-pixel counts on the same BEV source.
  - Streamlit restarted on `heracleum`; remote PID: `909437`.
  - Local HTTP check: `200`.
  - Browser URL: `http://127.0.0.1:8501`

## Occupancy Preview Live Update Fix

Date: 2026-06-28

- Problem:
  - Moving the Occupancy tab sliders did not visibly change the preview image.
  - Root cause: the tab displayed `last_run["paths"]["final_occupancy_grid"]`, a static PNG exported by the previous pipeline run.
  - The slider values were written to the in-memory config, but the page did not recompute obstacles, inflation, or occupancy for preview.
- Fix:
  - Added live source inputs for `bev_rgb.png`, `aligned_point_cloud.ply`, and `map_metadata.yaml`.
  - The Occupancy tab now recomputes semantic classes from the current BEV source and current Semantic config.
  - Raw obstacle cells are recomputed from the aligned point cloud with the current `Non-ground height threshold`.
  - Obstacles are inflated with the current `robot_radius + safety_margin`.
  - The final occupancy grid is fused live with the current `unknown_as_occupied` setting.
  - The page displays raw obstacles, inflated obstacles, live final occupancy, and counts for each stage.
- Troubleshooting note:
  - Current `outputs/track_map` aligned point cloud has `z_max` around `0.047 m`.
  - Therefore thresholds at or above `0.06 m` naturally produce zero height-based obstacle cells.
  - Useful threshold range for the current reconstruction is roughly `0.01-0.03 m`.
- Verification:
  - `.conda-vggt/bin/python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `.conda-vggt/bin/python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`
  - Direct checks on `outputs/track_map` showed raw obstacle cell counts change across `0.01/0.02/0.03 m` thresholds.
  - Streamlit restarted on `heracleum`; remote PID: `917647`.
  - Local HTTP check: `200`.
  - Browser URL: `http://127.0.0.1:8501`

## Remove Redundant Ground Texture Tab

Date: 2026-06-29

- Problem:
  - The standalone `Ground Texture` tab duplicated the IPM texture generation already needed in the Alignment step.
  - This made the UI look like the pipeline had to rebuild the same BEV texture later as a separate stage.
- Fix:
  - Removed the standalone `Ground Texture` tab from the top navigation.
  - Moved the IPM texture controls into the Alignment tab under `IPM texture settings`.
  - Alignment remains the canonical place where the VGGT camera-guided ground texture is generated for point picking.
  - The Alignment-generated texture and metadata are now preferred by Semantic and Occupancy live previews when present.
  - The full export config still keeps `ground_texture.enabled`, but the user-facing control now lives in Alignment instead of a separate page.
- Verification:
  - `.conda-vggt/bin/python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `.conda-vggt/bin/python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`
  - Streamlit restarted on `heracleum`; remote PID: `2171997`.
  - Local HTTP check: `200`.
  - Browser URL: `http://127.0.0.1:8501`

## BEV Metric Aligned Map View

Date: 2026-06-29

- Request:
  - The BEV page should show the aligned map using a fixed metric display convention.
  - Display convention: +y points upward, +x points leftward, 1000 pixels represent 1 m, and the local display origin is at the lower-right pixel.
- Troubleshooting route:
  - Checked the existing BEV tab and found it only displayed the raw raster/texture image with Streamlit's default image scaling.
  - Checked map metadata formats and found two active schemas: `map_metadata.yaml` stores bounds at top level, while Alignment IPM texture metadata stores them under `metadata`.
  - Added metadata parsing for both schemas so the BEV view can use either the rasterized point-cloud map or the Alignment-generated IPM texture.
  - Avoided clipping negative map coordinates by rendering the full map extent and treating the lower-right of the rendered image as the local display origin.
  - Used pixel-center resampling to keep the 1000 px/m display from gaining a one-pixel border or half-pixel offset.
- Fix:
  - Added `Metric Aligned Map` to the BEV tab.
  - The view loads the current aligned BEV image plus aligned map metadata, resamples it to 1000 px/m, and displays it with +x left / +y up.
  - Added optional 1 m x/y axes from the lower-right origin.
  - The page reports image size, origin pixel, lower-right world coordinate, upper-left world coordinate, and source file paths.
- Verification:
  - `.conda-vggt/bin/python -m py_compile duckietown_offline_mapper/app.py`: passed
  - `.conda-vggt/bin/python -m pytest -q duckietown_offline_mapper/tests`: `8 passed`
  - Streamlit restarted on `heracleum`; remote PID: `2286736`.
  - Local HTTP check: `200`.
  - Browser URL: `http://127.0.0.1:8501`
