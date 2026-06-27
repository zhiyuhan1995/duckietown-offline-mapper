Build a Python project named `duckietown_offline_mapper`.

Goal:
Create an offline mapping pipeline for Duckietown. The system takes either a monocular video or a folder of images as input and outputs a metric BEV semantic/occupancy grid suitable for path planning.

Core requirement:
The reconstruction path must use VGGT-SfM only. There should be no IPM or dummy reconstruction fallback in the final pipeline.

Updated high-level pipeline:
1. Load input video or images.
2. Extract keyframes if the input is a video.
3. Run VGGT-SfM to recover:
   - camera extrinsics and intrinsics
   - depth maps / point maps
   - a global colored point cloud
   - confidence values
   - optional COLMAP-format sparse model
4. Bootstrap the map frame:
   - fit a coarse dominant ground plane from the VGGT point cloud with RANSAC
   - rotate the reconstruction so the coarse ground plane lies at z=0
   - use this bootstrap plane only to define a stable top-down coordinate system
5. Run Level-2 3D Gaussian Splatting:
   - initialize training from VGGT-COLMAP cameras and points
   - train/optimize a 3D Gaussian scene representation from the selected keyframes
   - keep training and rendering isolated from the Streamlit service process
   - run GPU work on a remote GPU host, currently cluster-gpu03
6. Render Gaussian-derived BEV products:
   - orthographic top-down BEV RGB texture
   - alpha / coverage map
   - optional depth, height, and normal maps
   - optional debug trajectory and Gaussian point visualization
7. Use the Gaussian-rendered BEV texture, alpha, and height products as the preferred downstream map representation.
8. Refine / confirm the ground plane and metric alignment:
   - source control points are selected on the BEV plane
   - selected source z is always 0
   - target points are manually entered in real-world map-frame xy coordinates
   - estimate Sim(2) by default, with optional SE(2)
9. Crop the scene to the Duckietown track region using the aligned Gaussian BEV products.
10. Run BEV semantic segmentation on the Gaussian-rendered texture:
    - drivable road
    - non-drivable island / outside-track area
    - lane markings
    - stop lines
    - unknown
11. Generate a geometric obstacle occupancy layer:
    - use non-ground geometry and/or Gaussian-rendered height/depth products
    - project obstacles above a configurable height threshold into BEV
12. Fuse semantic non-drivable regions and geometric obstacle regions into a final occupancy map.
13. Inflate blocked cells according to the Duckiebot radius and safety margin.
14. Export ROS-compatible map products and project metadata.

Gaussian Splatting requirements:
- Follow the official VGGT recommendation:
  - export VGGT predictions to COLMAP format
  - train Gaussian Splatting from the generated `images/` and `sparse/` scene directory
- Use `gsplat` as the first target implementation.
- The first integration can be a standalone experiment script before it is wired into the full pipeline.
- Intermediate rendered products must be saved for inspection:
  - `gaussian_bev_rgb.png`
  - `gaussian_bev_alpha.png`
  - `gaussian_bev_height.png` or `gaussian_bev_depth.png` when available
  - trainer logs / metrics
  - a preview image visible from the Streamlit app
- The raw point-cloud BEV raster should remain available only as a debug comparison, not as the main semantic input once Gaussian BEV is enabled.

Interface:
Create a Streamlit UI with the following tabs:
1. Input:
   - upload video or select image folder
   - choose keyframe interval
   - preview keyframes
2. Reconstruction:
   - run VGGT-SfM reconstruction
   - load existing point clouds and Gaussian products
   - preview point cloud statistics
   - inspect the reconstructed point cloud with an interactive mouse-rotatable 3D viewer
3. Ground Plane:
   - run bootstrap RANSAC ground-plane fitting
   - sliders for distance threshold and max iterations
   - preview ground vs non-ground point counts
4. Gaussian Splatting:
   - prepare VGGT-COLMAP scene data
   - launch / monitor 3DGS training on the remote GPU host
   - render top-down orthographic BEV texture products
   - preview Gaussian-rendered BEV RGB / alpha / height
5. Planar Metric Alignment / Map Frame Definition:
   - show the Gaussian-rendered BEV texture when available
   - fall back to point-cloud BEV only for debug
   - allow the user to click points directly on the 2D BEV plane
   - convert each click to source point `(x_recon, y_recon, z_recon=0)`
   - ask the user to manually enter the corresponding map-frame coordinate `(x_map, y_map, z_map=0)`
   - require at least 3 non-collinear point correspondences
   - estimate Sim(2), or SE(2) when scale is known
   - save transform and residuals to `map_metadata.yaml`
6. Crop / ROI:
   - allow rectangular ROI with x/y bounds
   - optionally allow polygon ROI
7. BEV:
   - set BEV resolution in meters per pixel
   - choose Gaussian BEV or raw point raster debug source
   - preview RGB / alpha / height layers
8. Semantic Segmentation:
   - primary input is the Gaussian-rendered BEV RGB texture
   - the current HSV/RGB threshold segmentation is acceptable as a first semantic backend
   - the design must allow replacing color thresholds with a learned BEV semantic segmentation model
9. Occupancy:
   - set non-ground height threshold
   - set robot radius and safety margin
   - generate obstacle occupancy and final occupancy grid
10. Export:
   - save all map products to an output directory

Implementation requirements:
- Use Python 3.10+.
- Use numpy, opencv-python, open3d, scipy, scikit-image, pyyaml, streamlit.
- Use PyTorch and VGGT for reconstruction.
- Use gsplat for the Level-2 Gaussian Splatting experiment/integration.
- Keep reconstruction, Gaussian optimization, BEV rendering, semantic segmentation, occupancy, and export as modular components.
- Do not run heavy GPU work inside the Streamlit service process.
- Use subprocesses or CLI entry points for VGGT and 3DGS jobs.
- Keep remote GPU execution explicit and logged.
- Use Open3D for point cloud loading, saving, RANSAC plane fitting, and basic geometry utilities.
- Use numpy arrays for grids.
- Use OpenCV for BEV image writing, simple threshold-based segmentation, morphology, and visualization utilities.
- Implement clean coordinate conversion functions:
  - `world_to_grid(x, y, metadata)`
  - `grid_to_world(u, v, metadata)`
- Implement and keep unit tests for:
  - Sim(2) / Sim(3) alignment utilities
  - world/grid coordinate conversion
  - occupancy fusion logic
  - Gaussian BEV metadata and click-to-world conversion when added
- Make the code modular and readable. Do not put everything in one script.

Suggested file structure:
```
duckietown_offline_mapper/
  app.py
  requirements.txt
  README.md
  configs/
    default.yaml
    gaussian_3dgs_smoke.yaml
  src/
    io_utils.py
    keyframes.py
    reconstruction.py
    pointcloud.py
    plane.py
    alignment.py
    gaussian.py
    gaussian_bev.py
    bev.py
    segmentation.py
    occupancy.py
    export.py
    visualization.py
  tools/
    prepare_gsplat_scene.py
    train_gsplat_scene.py
    render_gaussian_bev.py
  tests/
    test_alignment.py
    test_grid_coordinates.py
    test_occupancy.py
    test_gaussian_bev.py
```

Final output requirement:
The pipeline must export a ROS-compatible occupancy grid map that can be loaded in RViz and used for path planning through ROS map_server / ROS2 nav2_map_server.

The export must include:

1. `map.pgm` or `map.png`
   A 2D occupancy map image.
   Pixel convention:
   - free cells: white, value 254 or 255
   - occupied cells: black, value 0
   - unknown cells: gray, value 205

2. `map.yaml`
   A ROS map metadata file compatible with map_server / nav2_map_server.

Example:
```yaml
image: map.pgm
mode: trinary
resolution: 0.02
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

3. `occupancy_grid.npy`
   Raw numpy array with ROS OccupancyGrid values:
   - 0 = free
   - 100 = occupied
   - -1 = unknown

4. `map_metadata.yaml`
   Additional project metadata:
   - resolution
   - map frame name
   - grid width and height
   - x_min, x_max, y_min, y_max
   - world_to_grid convention
   - grid_to_world convention
   - reconstruction_to_map transform
   - bootstrap ground plane
   - Gaussian Splatting training/render metadata
   - robot radius
   - safety margin
   - obstacle inflation radius
   - semantic classes
   - BEV generation parameters
   - alignment control points and residual error

5. Gaussian BEV debug products:
   - `gaussian_bev_rgb.png`
   - `gaussian_bev_alpha.png`
   - `gaussian_bev_height.png` or `gaussian_bev_depth.png` when available
   - raw point-raster BEV comparison image
