# Story

Duckietown mapping from monocular video is attractive because the robot can build a planning map from ordinary camera recordings, without a lidar or motion-capture setup. VGGT-SfM gives us a strong first step: it can recover a global scene geometry, camera poses, and a colored point cloud from the video. That is enough to prove the coordinate system, estimate a ground plane, and build a first BEV occupancy grid.

The raw point-cloud BEV is not good enough as the long-term semantic input. It is a map made of samples, not a continuous surface. Road edges become ragged, lane markings break into dots, black road regions disappear in low-confidence areas, and the resulting semantic segmentation has to fight holes and jagged boundaries that were introduced by the reconstruction representation itself.

The next version should insert a 3D Gaussian Splatting stage after VGGT-SfM. The idea is to use VGGT as the bootstrap geometry and camera provider, then optimize a continuous radiance representation from the selected keyframes. Once the Gaussian scene is trained, we can render a top-down orthographic BEV texture. This BEV should look more like a continuous map image than a cloud of points.

The intended story becomes:

1. Use VGGT-SfM to recover camera poses, intrinsics, and initial geometry.
2. Use a coarse RANSAC ground plane only to define the initial BEV/map coordinate system.
3. Export VGGT predictions to a COLMAP-style scene.
4. Train 3D Gaussians from the images, cameras, and initial points.
5. Render a high-resolution top-down BEV texture, alpha/coverage, and height/depth layers.
6. Run semantic segmentation and occupancy fusion on the Gaussian-rendered BEV products.
7. Export a ROS-compatible occupancy grid for path planning.

This is a shift from "draw the points and threshold their colors" to "reconstruct a continuous map texture, then reason over that map." The visual quality matters because the map is not just for humans; clean, continuous BEV boundaries make the semantic and occupancy decisions less brittle.
