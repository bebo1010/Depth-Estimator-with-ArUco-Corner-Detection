# Depth Estimator with ArUco Corner Detection

## Functionality
- [x] Read two Realsense IR camera streams and Realsense depth stream
- [x] Detect ArUco markers for each stream
- [x] Compute depths with detected ArUco markers and get depth from Realsense depth stream

### buttons for depth_estimator_streaming.py

- `s` or `S` to save the images
- `esc` to close program

### buttons for depth_estimator_images.py

- `l` or `L` to load the images
- `n` or `N` to show next image set
- `p` or `P` to show previous image set
- `esc` to close program

## Goal
- [ ] Refactor the code to include interfaces for FLIR cameras
- [ ] Clean up the codes

| File | Usage |
| --- | --- |
| depth_estimator_streaming.py | Compute depths to ArUco from Realsense camera |
| depth_estimator_images.py | Load the results saved before |