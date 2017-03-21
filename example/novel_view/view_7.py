#!/usr/bin/python3
# ==============================================================
# viewpoint systhesis single plane-base
# ==============================================================
import numpy as np
import sys
import scipy.misc
from sklearn.decomposition import PCA
import triangle

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import dashcam_parse
import google_parse
import glumpy_setting
import base_process


sleIndex = 3
createSV = True
createSFM = False
needAlign = True
needMatchInfo3d = True
imageSynthesis = True
needGround = False
addPlane = False  # need open auto_plane in google_parse
needVisual = True
needTexture = False
mapType = '_trajectory'  # [_info3d, _trajectory]
randomPos = [5, 0, 0]
randomDeg = 0

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
"""
Process the select file
"""
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
    if createSV:
        sv3DRegion.create_topoloy()
        sv3DRegion.create_region_time(start=6, end=11)
        # sv3DRegion.create_region()
        pano_length = len(sv3DRegion.panoramaList)
        anchor_inv = np.linalg.inv(sv3DRegion.anchorMatrix)
        zero_vec, pano_ori_set, dis_len_set = [0, 0, 0, 1], np.zeros((pano_length, 3)), np.zeros(pano_length)
        # Initialize the pano according to location(lat, lon)
        for i in range(0, pano_length):
            sv3D = sv3DRegion.sv3D_Time[i]
            sv3D.apply_global_adjustment()  # Absolute position on earth (lat, lon, yaw)
            sv3D.apply_local_adjustment()  # Relative position according to anchor (anchor's lat,lon)
            if needMatchInfo3d:
                # This rotate the SV3D for matching x-y plane
                # Actually we build the point cloud on x-y plane
                # So we just multiply the inverse matrix of anchor
                sv3D.apply_anchor_adjustment(anchor_matrix=sv3DRegion.anchorMatrix)
            # Record all pano location(relative)
            # And find the nearest panorama
            if imageSynthesis:
                pano_ori = np.dot(zero_vec, sv3D.matrix_local)
                pano_ori = np.dot(pano_ori, anchor_inv)
                pano_ori_set[i] = pano_ori[:3]
                dis_vec = (pano_ori[:3]) - randomPos
                dis_len_set[i] = np.linalg.norm(dis_vec)
                if i == (pano_length-1):
                    nearest_pano_idx = np.argmin(dis_len_set)
                    syn_pano = np.zeros((256, 512, 3), dtype=np.float32)
        # Gather the pointcloud data (may be aligned or not)
        # Why separate in two loops?
        # Because there once existed a process about alignment here
        # And if we want to know which panorama is the closet(not necessary)
        for i in range(0, pano_length):
            sv3D = sv3DRegion.sv3D_Time[i]
            if i == 0:
                data = sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]
                dataGnd = sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]
            else:
                data = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]), axis=0)
                dataGnd = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]), axis=0)

            if imageSynthesis and i == nearest_pano_idx:
                # Visualize which panorama is the closet
                if needVisual:
                    dataNearest = np.zeros(pano_length + 1, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
                    dataNearest['a_color'] = [0, 1, 0]
                    dataNearest['a_color'][nearest_pano_idx] = [1, 1, 0]
                    dataNearest['a_position'][0:-1] = pano_ori_set
                    dataNearest['a_position'][-1] = randomPos
                    dataNearest['a_color'][-1] = [1, 0, 0]
                # calculate synthesis view
                ori_pano = scipy.misc.imresize(sv3D.panorama, (256, 512), interp='bilinear', mode=None) / 255.0
                index_pano = np.full((256, 512), False, dtype=bool)
                point = sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]['a_position']
                if needTexture:
                    texture_xyz, texture_color, texture_len = [], [], 0
                for point in sv3D.ptCLoudData:
                    if np.isnan(point['a_position']).any():
                        continue
                    x, y, z = point['a_position'] - np.array(randomPos)
                    lng, lat = base_process.pos_2_deg(x, y, z)

                    lng = (lng + randomDeg) % 360
                    img_x = int(lng / 360.0 * 512.0)
                    img_y = int(-(lat - 90) / 180.0 * 256.0)
                    syn_pano[img_y, img_x, :] = point['a_color']
                    index_pano[img_y, img_x] = True
                    if needTexture:
                        texture_xyz.append([img_x, img_y, 0])
                        texture_color.append(point['a_color'])
                        texture_len += 1
                if needTexture:
                    texture_data = np.zeros((texture_len), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
                    texture_data['a_color'] = np.array(texture_color)
                    texture_data['a_position'] = np.array(texture_xyz)

                    #pca = PCA(n_components=2)
                    #data_transfer = pca.fit_transform(texture_data['a_position'])
                    data_transfer = texture_data['a_position'][:, 0:2]
                    texture_tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
                    texture_data['a_position'][:, 0] = texture_data['a_position'][:, 0] / 256.0 - 1.0
                    texture_data['a_position'][:, 1] = texture_data['a_position'][:, 1] / 128.0 - 1.0



        #show_pano = np.concatenate((ori_pano, syn_pano), axis=1)
        #scipy.misc.imsave('yo.png', syn_pano)
        #scipy.misc.imshow(plane_pano)
        if needVisual:
            programSV3DRegion = glumpy_setting.ProgramSV3DRegion(
                data=data, name='programSV3DRegion',
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
            programSV3DRegionGnd = glumpy_setting.ProgramSV3DRegion(
                data=dataGnd, name='programSV3DRegionGnd',
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
            programSV3DTopology = glumpy_setting.ProgramSV3DTopology(
                data=sv3DRegion.topologyData, name='programSV3DTopology',
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
            programSV3DNearest = glumpy_setting.ProgramSV3DTopology(
                data=dataNearest, name='programSV3DNearest',
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d,
                point_size=20, alpha=0.6)

        '''
        import triangle
        data = np.zeros((256, 512), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        data['a_color'] = ori_pano
        for img_y in range(0, 256):
            for img_x in range(0, 512):
                data[img_y, img_x]['a_position'] = [img_y, img_x, 0]
        data_y = np.reshape(data['a_position'], (256*512, 3))
        from sklearn.decomposition import PCA
        import triangle
        pca = PCA(n_components=2)
        data_transfer = pca.fit_transform(data_y)
        tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
        programImage = glumpy_setting.ProgramPlane(data=data, name='test',
                                                    face=tri)
        '''


"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()
    if addPlane:
        for j in range(0, pano_length):
            sv3D = sv3DRegion.sv3D_Time[j]
            for i in range(0, len(sv3D.gnd_plane)):
                programGround = glumpy_setting.ProgramPlane(data=sv3D.gnd_plane[i]['data'], name='test',
                                                            face=sv3D.gnd_plane[i]['tri'])
                gpyWindow.add_program(programGround)

    if createSFM:
        gpyWindow.add_program(programSFM3DRegion)
        gpyWindow.add_program(programTrajectory)

    if createSV:
        gpyWindow.add_program(programSV3DRegion)
        gpyWindow.add_program(programSV3DTopology)
        if imageSynthesis:
            gpyWindow.add_program(programSV3DNearest)
            #gpyWindow.add_program(programImage)
        if needGround:
            gpyWindow.add_program(programSV3DRegionGnd)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()

if needTexture:
    gpyViewWindow = glumpy_setting.GpyViewWindow(texture_data, texture_tri)
    gpyViewWindow.run()