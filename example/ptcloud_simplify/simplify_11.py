#!/usr/bin/python3
# ==============================================================
# match all panos
# now breing back the triangle
# ==============================================================
import numpy as np
import sys
import scipy.misc
import cv2
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import dashcam_parse
import google_parse
import glumpy_setting
import base_process


sleIndex = 3
addPlane = True
createSV = True
createSFM = False
needAlign = True
needMatchInfo3d = True
needVisual = True
needGround = False
mapType = '_trajectory' # [_info3d, _trajectory]

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
"""
Process the select file
"""

class LinearTranslation():
    def fit(self, X, y, sample_weight=None):
        self.coef_ = np.mean(y - X, axis=0)

    def score(self, X, y, sample_weight=None):
        y_ = X + self.coef_
        return mean_squared_error(y, y_)


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

for fileIndex in range(sleIndex, sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    """
    Create the sfm point cloud
    """
    sfm3DRegion = dashcam_parse.SFM3DRegion(fileID)
    if createSFM:
        programSFM3DRegion = glumpy_setting.ProgramSFM3DRegion(data=sfm3DRegion.ptcloudData, name='ProgramSFM3DRegion',
                                                               point_size=1, matrix=sfm3DRegion.matrix)
        programTrajectory = glumpy_setting.ProgramTrajectory(data=sfm3DRegion.trajectoryData, name='programTrajectory',
                                                             point_size=5, matrix=sfm3DRegion.matrix)
        if needAlign:
            programSFM3DRegion.align_flip()
            programTrajectory.align_flip()
    """
    Create the global metric point cloud,
    then set the region anchor
    """
    # anchor is sorted and pick the first one
    # now load the anchor that had been stored
    anchor = dashCamFileProcess.get_trajectory_anchor(fileID)
    fileID += mapType
    sv3DRegion = google_parse.StreetView3DRegion(fileID)
    sv3DRegion.init_region(anchor=anchor)
    # why QQ?
    # because the map didn't query the anchor file
    if sv3DRegion.QQ:
        continue

    if createSV:
        index = 0
        sv3DRegion.create_topoloy()
        sv3DRegion.create_region_time(start=10, end=11)
        length = len(sv3DRegion.panoramaList)
        # Initialize the pano according to location(lat, lon)
        for i in range(0, length):
            sv3D = sv3DRegion.sv3D_Time[i]
            sv3D.apply_global_adjustment()  # Absolute position on earth (lat, lon, yaw)
            sv3D.apply_local_adjustment()  # Relative position according to anchor (anchor's lat,lon)
            #sv3D.apply_anchor_adjustment(anchor_matrix=sv3DRegion.anchorMatrix)
        # Find matched feature point
        match_pair_all = []
        for i in range(0, length-1):
            print('[{:d}-{:d}]/{:d}'.format(i, i+1, len(sv3DRegion.panoramaList)-1))
            panorama_a = sv3DRegion.panoramaList[i]  # queryImage
            panorama_b = sv3DRegion.panoramaList[i+1]  # trainImage
            sv3D_a = sv3DRegion.sv3D_Time[i].ptCLoudData['a_position']
            sv3D_b = sv3DRegion.sv3D_Time[i+1].ptCLoudData['a_position']
            #panorama_a = cv2.resize(panorama_a, (512, 256))
            #panorama_b = cv2.resize(panorama_b, (512, 256))

            # Initiate SIFT detector [SIFT, SURF, ORB]
            orb = cv2.ORB_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = orb.detectAndCompute(panorama_a, None)
            kp2, des2 = orb.detectAndCompute(panorama_b, None)

            matcher = cv2.BFMatcher()

            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            match_pair = []
            RANSACResult = []
            matchX1, matchX2 = [], []
            for mat in matches:
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt
                x1 = int(x1 * 512. / 832.)
                y1 = int(y1 * 256. / 416.)
                x2 = int(x2 * 512. / 832.)
                y2 = int(y2 * 256. / 416.)
                xyz1, xyz2 = sv3D_a[y1*512 + x1], sv3D_b[y2*512 + x2]
                if np.isnan(xyz1).any() or np.isnan(xyz2).any():
                    continue
                matchX1.append(xyz1)
                matchX2.append(xyz2)
                if np.linalg.norm(xyz1 - xyz2) > 1:
                    continue
                match_pair.append(xyz2 - xyz1)
                #end_img = drawMatches(panorama_a, kp1, panorama_b, kp2, [mat])
                #scipy.misc.imshow(end_img)
            if len(match_pair) == 0:
                match_pair_all.append([0, 0, 0])
            if len(matchX1) == 0:
                RANSACResult.append([0, 0, 0])
            else:
                match_pair_all.append(np.mean(match_pair, axis=0))
                model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
                model_ransac.fit(np.array(matchX1), np.array(matchX2))
                inlier_mask = model_ransac.inlier_mask_
                RANSACResult.append(model_ransac.estimator_.coef_)

        # Minimize the error
        for i in range(0, length):
            sv3D = sv3DRegion.sv3D_Time[i]
            #sv3D.auto_plane()
            if index == 0:
                 data = sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]
                 dataGnd = sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]
            else:
                 print(match_pair_all[i-1])
                 #print(RANSACResult[i-1])
                 for ii in range(1, i+1):
                    sv3D.ptCLoudData['a_position'] -= match_pair_all[ii-1]
                 data = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]), axis=0)
                 dataGnd = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]), axis=0)
            sv3D.auto_plane()

            index += 1

        programSV3DRegion = glumpy_setting.ProgramSV3DRegion(
            data=data, name='ProgramSV3DRegion',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw)
        programSV3DRegionGnd = glumpy_setting.ProgramSV3DRegion(
            data=dataGnd, name='ProgramSV3DRegion',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw)
        programSV3DTopology = glumpy_setting.ProgramSV3DTopology(
            data=sv3DRegion.topologyData, name='ProgramSV3DRegion',
            anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw)

        if needMatchInfo3d:
            # This rotate the SV3D for matching x-y plane
            # Actually we build the point cloud on x-y plane
            # So we just multiply the inverse matrix of anchor
            programSV3DRegion.apply_anchor_flip()
            programSV3DRegionGnd.apply_anchor_flip()
            programSV3DTopology.apply_anchor_flip()

"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()
    if addPlane:
        for j in range(0, length):
            sv3D = sv3DRegion.sv3D_Time[j]
            for i in range(0, len(sv3D.gnd_plane)):
                programGround = glumpy_setting.ProgramPlane(data=sv3D.gnd_plane[i]['data'], name='test',
                                                            face=sv3D.gnd_plane[i]['tri'])
                gpyWindow.add_program(programGround)

    if createSFM:
        gpyWindow.add_program(programSFM3DRegion)
        gpyWindow.add_program(programTrajectory)

    if createSV:
        #gpyWindow.add_program(programSV3DRegion)
        #gpyWindow.add_program(programSV3DTopology)
        if needGround:
            gpyWindow.add_program(programSV3DRegionGnd)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()
