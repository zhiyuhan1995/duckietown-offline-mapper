Build a Python project named duckietown_offline_mapper.

Goal:
Create an offline mapping pipeline for Duckietown. The system takes either a monocular video or a folder of images as input and outputs a metric BEV semantic/occupancy grid suitable for path planning.

High-level pipeline:
1. Load input video or images.
2. Extract keyframes if the input is a video.
3. Run a reconstruction backend, initially using a placeholder interface for VGGT. The backend should output:
   - camera poses
   - camera intrinsics
   - dense depth maps or a global colored point cloud
   - confidence values if available
4. Fuse the reconstructed points into a global colored point cloud.
5. Estimate the dominant ground plane with RANSAC.
6. Align the reconstruction to a user-defined metric map frame using at least three control points. Implement Umeyama / Procrustes Sim(3) alignment.
7. Crop the scene to the Duckietown track region.
8. Rasterize the aligned colored point cloud onto the ground plane to generate a global BEV RGB texture map.
9. Segment the BEV RGB map into semantic classes:
   - drivable road
   - non-drivable island / outside-track area
   - lane markings
   - stop lines
   - unknown
   The first implementation can use color thresholds and morphology operations.
10. Generate a geometric obstacle occupancy layer by projecting non-ground points above a configurable height threshold into BEV.
11. Fuse semantic non-drivable regions and geometric obstacle regions into a final occupancy map.
12. Inflate blocked cells according to the Duckiebot radius.
13. Export:
   - aligned_point_cloud.ply
   - bev_rgb.png
   - semantic_mask.png
   - obstacle_occupancy.png
   - final_occupancy_grid.png
   - semantic_grid.npy
   - obstacle_grid.npy
   - occupancy_grid.npy
   - map_metadata.yaml

Interface:
Create a Streamlit UI with the following tabs:
1. Input:
   - upload video or select image folder
   - choose keyframe interval
   - preview keyframes
2. Reconstruction:
   - run reconstruction backend
   - load existing point cloud if reconstruction is already done
   - preview basic point cloud statistics
3. Ground Plane:
   - run RANSAC ground-plane fitting
   - sliders for distance threshold and max iterations
   - preview ground vs non-ground point counts
4. Planar Metric Alignment / Map Frame Definition:
   - Show the preliminary BEV RGB map generated from the ground-aligned reconstructed point cloud.
   - Allow the user to click points directly on the 2D BEV image.
   - For each clicked point, convert the pixel location back to its reconstruction-plane coordinate:
       source point = (x_recon, y_recon, z_recon=0)
   - Ask the user to manually enter the corresponding real-world map-frame coordinate:
       target point = (x_map, y_map, z_map=0)
   - Require at least 3 non-collinear point correspondences.
   - Estimate a 2D similarity transform Sim(2):
       [x_map, y_map]^T = s * R(theta) * [x_recon, y_recon]^T + t
   - If the scale is already known, optionally estimate only an SE(2) transform:
       [x_map, y_map]^T = R(theta) * [x_recon, y_recon]^T + t
   - Apply the estimated transform to:
       1. the aligned point cloud
       2. the BEV RGB map metadata
       3. the semantic grid
       4. the occupancy grid
   - Save the transform and map metadata to map_metadata.yaml.
   
   The user should not need to type reconstructed 3D coordinates manually. They should only click points on the BEV image and enter the corresponding real-world (x, y) coordinates. All clicked control points are assumed to lie on the ground plane with z=0.
5. Crop / ROI:
   - allow user to set x_min, x_max, y_min, y_max
   - optionally allow polygon ROI
6. BEV:
   - set BEV resolution in meters per pixel
   - rasterize point cloud to BEV RGB image
   - preview result
7. Semantic Segmentation:
   - provide HSV/RGB threshold sliders for black road, white lines, yellow lines, red stop lines
   - provide morphological closing / opening parameters
   - output semantic mask
8. Occupancy:
   - set non-ground height threshold
   - set robot radius and safety margin
   - generate obstacle occupancy and final occupancy grid
9. Export:
   - save all outputs to an output directory

Implementation requirements:
- Use Python 3.10+.
- Use numpy, opencv-python, open3d, scipy, scikit-image, pyyaml, streamlit.
- Keep the VGGT part modular. Create a class ReconstructionBackend with a method run(input_path, output_dir). For now, implement a dummy backend that can load an existing point_cloud.ply and camera data. Leave clear TODO hooks for integrating VGGT.
- Use Open3D for point cloud loading, saving, RANSAC plane fitting, and basic visualization.
- Use numpy arrays for grids.
- Use OpenCV for BEV image creation, threshold-based segmentation, and morphology.
- Implement clean coordinate conversion functions:
  - world_to_grid(x, y, metadata)
  - grid_to_world(u, v, metadata)
- Implement unit tests for:
  - Sim(3) alignment
  - world/grid coordinate conversion
  - occupancy fusion logic
- Make the code modular and readable. Do not put everything in one script.

Suggested file structure:
duckietown_offline_mapper/
  app.py
  requirements.txt
  README.md
  configs/
    default.yaml
  src/
    io_utils.py
    keyframes.py
    reconstruction.py
    pointcloud.py
    plane.py
    alignment.py
    bev.py
    segmentation.py
    occupancy.py
    export.py
    visualization.py
  tests/
    test_alignment.py
    test_grid_coordinates.py
    test_occupancy.py
    
Final output requirement:

The pipeline must export a ROS-compatible occupancy grid map that can be loaded in RViz and used for path planning through ROS map_server / ROS2 nav2_map_server.

The export must include:

1. map.pgm or map.png
   A 2D occupancy map image.
   Pixel convention:
   - free cells: white, value 254 or 255
   - occupied cells: black, value 0
   - unknown cells: gray, value 205

2. map.yaml
   A ROS map metadata file compatible with map_server / nav2_map_server.

Example:
image: map.pgm
mode: trinary
resolution: 0.02
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196

3. occupancy_grid.npy
   Raw numpy array with ROS OccupancyGrid values:
   - 0 = free
   - 100 = occupied
   - -1 = unknown

4. map_metadata.yaml
   Additional project metadata:
   - resolution
   - map frame name
   - grid width and height
   - x_min, x_max, y_min, y_max
   - world_to_grid convention
   - grid_to_world convention
   - reconstruction_to_map transform
   - robot radius
   - safety margin
   - obstacle inflation radius
   - semantic classes
   - BEV generation parameters
   - alignment control points and residual error
