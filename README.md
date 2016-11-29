# DashCam_python
A project about dashcam implemented by Python

## Current pipeline
0. Fetch the Google data (Use info_3d)
1. Transform the point in SFM to Google (SV3D)
2. Align the SFM with Google (By sfm3DRegion.matrix)
3. Output the trajectory in (lat, lon, h)/(x, y, z) form
------------------------------------------------------------
4. Fetch the Google data (Use trajectory)
5. Create the SV3D according to trajectory(Now, it's important to use the correct anchor!!)
6. Output the ply file of SV3D constructed from trajectory

- ![3D2G](/src/CITYSCAPES_DCGAN_3D2G/3D2G.gif)
- TODO:
  0. Fetch the Google data (file lacked anchor)
