# DashCam_python
A project about dashcam implemented by Python
[paper on eccv'16 workshop](http://link.springer.com/chapter/10.1007/978-3-319-46604-0_10)
[intro video](https://www.youtube.com/watch?v=qeIMMk8E17o)

## Current pipeline
1. Fetch the Google data (Use info_3d)
2. Transform the point in SFM to Google (SV3D)
3. Align the SFM with Google (By sfm3DRegion.matrix)
4. Output the trajectory in (lat, lon, h)/(x, y, z) form
5. Fetch the Google data (Use trajectory)
6. Create the SV3D according to trajectory(Now, it's important to use the correct anchor!!)
7. Output the ply file of SV3D constructed from trajectory

- ![3D2G](/src/CITYSCAPES_DCGAN_3D2G/3D2G.gif)
- TODO:
  0. Fetch the Google data (file lacked anchor)
