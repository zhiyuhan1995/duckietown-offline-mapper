# duckietown_offline_mapper

Offline Semantic-Occupancy BEV Mapping for Duckietown from monocular video, using VGGT-SfM only.

The reconstruction stage is no longer an IPM fallback. The only supported backend is `vggt_sfm`: VGGT predicts camera parameters and dense depth/point geometry, the backend fuses confident points into a global colored point cloud, optionally exports a COLMAP-compatible scene for standalone 3DGS QA, then the mapping stack fits the bootstrap ground plane, aligns the BEV map frame, segments drivable semantics from the point-cloud BEV raster, projects non-ground obstacle points, inflates obstacles, and exports a ROS-compatible occupancy grid.

The 3D Gaussian Splatting work is currently a standalone quality-gate experiment. Gaussian renders are not consumed by semantic segmentation or occupancy export until a realistic camera-view and BEV result passes visual QA.

## PULSATILLA GPU Setup

```bash
ssh pulsatilla
cd /home/hanzhiyu/projects/duckietown
conda create -y -p .conda-vggt python=3.10 pip
.conda-vggt/bin/python -m pip install --upgrade pip
.conda-vggt/bin/python -m pip install -r duckietown_offline_mapper/requirements.txt
```

This environment was tested on pulsatilla with RTX 4090 GPUs.

## Run VGGT-SfM Pipeline

Default config samples `track.mp4` uniformly for VGGT:

```bash
ssh pulsatilla
cd /home/hanzhiyu/projects/duckietown
CUDA_VISIBLE_DEVICES=0 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py
```

Custom run:

```bash
CUDA_VISIBLE_DEVICES=0 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py \
  --input track.mp4 \
  --output outputs/track_map \
  --keyframe-interval 120 \
  --max-keyframes 12 \
  --resolution 0.02
```

VGGT weights are downloaded from Hugging Face on first run. Setting `HF_TOKEN` can improve rate limits.

## Standalone 3DGS QA

The 3DGS stage is isolated in `.venv-gsplat` so the gsplat example `pycolmap.SceneManager` dependency does not conflict with the VGGT environment.

Patch the local gsplat example when running train-all experiments with `--test_every 0`:

```bash
.conda-vggt/bin/python duckietown_offline_mapper/tools/patch_gsplat_no_holdout.py
```

Generate the small VGGT seed scene first:

```bash
CUDA_VISIBLE_DEVICES=1 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py \
  --config duckietown_offline_mapper/configs/3dgs_qa_dense_seed_scene.yaml
```

The mapper can export a small VGGT-COLMAP seed scene, but dense Gaussian training frames should be prepared separately. For `track.mp4`, every third frame gives 508 training images:

```bash
cd /home/hanzhiyu/projects/duckietown
.conda-vggt/bin/python duckietown_offline_mapper/tools/prepare_every3_gaussian_scene.py \
  --video track.mp4 \
  --output-dir outputs/track_map_3dgs_qa_every3_colmap_scene \
  --interval 3
```

Align the dense COLMAP scene back to the VGGT seed coordinate frame before training 3DGS:

```bash
.conda-vggt/bin/python duckietown_offline_mapper/tools/align_colmap_scene_to_vggt.py \
  --source-scene outputs/track_map_3dgs_qa_every3_colmap_scene \
  --source-sparse outputs/track_map_3dgs_qa_every3_colmap_scene/sparse_fixed/0 \
  --target-sparse outputs/track_map_3dgs_qa_dense_scene/work/gsplat_scene/sparse \
  --output-scene outputs/track_map_3dgs_qa_every3_aligned_vggt_scene
```

Standalone QA render products are inspected from the `3DGS QA` UI tab:

- camera-view contact sheets from `tools/render_gsplat_camera_views.py`
- plane-projected BEV previews from `tools/render_gaussian_plane_bev.py`
- orthographic exploratory renders from `tools/render_gaussian_bev.py`

These files are debug artifacts only; `bev_rgb.png` in a normal mapper export remains the point-cloud BEV raster.

## Bundle Adjustment

Direct COLMAP export is enabled by default through `reconstruction.sfm.save_colmap: true`.

For a small BA test:

```bash
CUDA_VISIBLE_DEVICES=0 .conda-vggt/bin/python duckietown_offline_mapper/run_pipeline.py \
  --config duckietown_offline_mapper/configs/pulsatilla_vggt_ba_smoke.yaml
```

BA uses the VGGT/VGGSfM tracker, LightGlue dependencies, and pycolmap. It is slower than direct VGGT depth fusion.

## Run UI

```bash
ssh -L 8501:localhost:8501 pulsatilla
cd /home/hanzhiyu/projects/duckietown
CUDA_VISIBLE_DEVICES=0 .conda-vggt/bin/python -m streamlit run duckietown_offline_mapper/app.py \
  --server.port 8501
```

The UI exposes the requested tabs: Input, Reconstruction, Ground Plane, 3DGS QA, Alignment, Crop / ROI, BEV, Semantic, Occupancy, and Export.

## Outputs

The export directory contains:

- `aligned_point_cloud.ply`
- `bev_rgb.png`
- `semantic_mask.png`
- `obstacle_occupancy.png`
- `final_occupancy_grid.png`
- `semantic_grid.npy`
- `obstacle_grid.npy`
- `occupancy_grid.npy`
- `map.png`
- `map.yaml`
- `map_metadata.yaml`
- `run_summary.yaml`
- `work/camera_extrinsics.npy`
- `work/camera_intrinsics.npy`
- `work/vggt_point_cloud.ply`
- `work/colmap_sparse/` when COLMAP export is enabled
- `work/gsplat_scene/` when COLMAP export is enabled for standalone 3DGS QA

`map.yaml` is compatible with ROS `map_server` and ROS2 `nav2_map_server`.

Occupancy conventions:

- `occupancy_grid.npy`: `0 = free`, `100 = occupied`, `-1 = unknown`
- `map.png`: free `254`, occupied `0`, unknown `205`

## Coordinate Flow

1. VGGT-SfM predicts cameras and depth/point geometry in a reconstruction coordinate frame.
2. Open3D RANSAC estimates the dominant ground plane.
3. The cloud is rotated/translated so the ground plane is `z=0`.
4. Optional user control points estimate Sim(2)/SE(2) metric map alignment.
5. Auto or manual ROI defines BEV bounds.
6. BEV semantic segmentation uses the point-cloud BEV raster until the standalone 3DGS quality gate is passed.
7. Semantic non-drivable regions and non-ground obstacle projection fuse into ROS occupancy.
