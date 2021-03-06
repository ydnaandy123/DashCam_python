#!/usr/bin/python3
# ==============================================================
# viewpoint systhesis with planeJoints and continuous
# ==============================================================
import numpy as np
import scipy.misc
from sklearn.decomposition import PCA
import triangle
import scipy.ndimage

import sys
sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import dashcam_parse
import google_parse
import glumpy_setting
import base_process


sleIndex = 3
createSFM = False
createSV = True
needAlign = True
needMatchInfo3d = True
needGround = False
addPlane = False
planeJoints = False
needVisual = True
needPerspective = False
imageSynthesis = True
needTexture = False
mapType = '_trajectory'  # [_info3d, _trajectory]
offsetPos = [0.2569, 7.7, 0]
offsetDeg = 0
texture_height, texture_width = 256.0, 512.0

# Create dashCamFileProcess and load 50 top Dashcam
dashCamFileProcess = file_process.DashCamFileProcessor()
#for yo in range(0, 10):
#    offsetPos = [0.367 + yo*(0.616 - 0.367)/10, 11.0 + yo*(18.0 - 11.0)/10.0, 0]
#    print(offsetPos)
"""
Process the select file
"""
for fileIndex in range(sleIndex, sleIndex+1):
    fileID = str(dashCamFileProcess.list50[fileIndex][1])
    print(fileID, fileIndex)
    """
    Create the sfm point cloud
    """
    if createSFM:
        sfm3DRegion = dashcam_parse.SFM3DRegion(fileID)
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
        sv3DRegion.create_region_time(start=8, end=11)
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
                dis_vec = (pano_ori[:3]) - offsetPos
                dis_len_set[i] = np.linalg.norm(dis_vec)
                if i == (pano_length-1):
                    nearest_pano_idx = np.argmin(dis_len_set)
        # Gather the pointcloud data (may be aligned or not)
        # Why separate in two loops?
        # Because there once existed a process about alignment here
        # And if we want to know which panorama is the closet(not necessary)
        for i in range(0, pano_length):
            sv3D = sv3DRegion.sv3D_Time[i]
            if addPlane:
                sv3D.auto_plane()
            if i == 0:
                data = sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]
                dataGnd = sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]
            else:
                data = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.non_con)]), axis=0)
                dataGnd = np.concatenate((data, sv3D.ptCLoudData[np.nonzero(sv3D.gnd_con)]), axis=0)

            if needPerspective:
                ori_pano = sv3D.panorama / 255.0
                pano_height, pano_width = ori_pano.shape[0], ori_pano.shape[1]  # Actually, this must be 1:2
                perspective_height, perspective_width = int(pano_height/4), int(pano_width/4)
                perspective_90_set = []
                #randomDeg = - sv3D.yaw
                for degree in range(0, 360, 90):
                    perspective_90 = np.zeros((perspective_height, perspective_width, 3))
                    for p_y in range(0, perspective_height):
                        for p_x in range(0, perspective_width):
                            x = p_x - perspective_width/2
                            z = -p_y + perspective_height/2
                            y = perspective_height
                            lng, lat = base_process.pos_2_deg(x, y, z)

                            lng = (lng + degree + offsetDeg) % 360
                            img_x = lng / 360.0 * pano_width
                            img_y = -(lat - 90) / 180.0 * pano_height

                            img_pos0_x = np.floor(img_x)
                            img_pos0_y = np.floor(img_y)

                            img_pos_diff_x = img_x - img_pos0_x
                            img_pos_diff_y = img_y - img_pos0_y

                            img_pos1_x = img_pos0_x + 1
                            img_pos1_y = img_pos0_y

                            img_pos2_x = img_pos0_x
                            img_pos2_y = img_pos0_y + 1

                            img_pos3_x = img_pos0_x + 1
                            img_pos3_y = img_pos0_y + 1

                            if img_pos1_x == pano_width:
                                img_pos1_x = pano_width - 1
                            if img_pos3_x == pano_width:
                                img_pos3_x = pano_width - 1
                            if img_pos2_y == pano_height:
                                img_pos2_y = pano_height - 1
                            if img_pos3_y == pano_height:
                                img_pos3_y = pano_height - 1

                            img_ratio0 = (1-img_pos_diff_x) * (1-img_pos_diff_y)
                            img_ratio1 = img_pos_diff_x * (1-img_pos_diff_y)
                            img_ratio2 = (1-img_pos_diff_x) * img_pos_diff_y
                            img_ratio3 = img_pos_diff_x * img_pos_diff_y

                            img_color0 = ori_pano[img_pos0_y, img_pos0_x, :]
                            img_color1 = ori_pano[img_pos1_y, img_pos1_x, :]
                            img_color2 = ori_pano[img_pos2_y, img_pos2_x, :]
                            img_color3 = ori_pano[img_pos3_y, img_pos3_x, :]

                            img_color = img_ratio0*img_color0 + img_ratio1*img_color1 + \
                                        img_ratio2*img_color2 + img_ratio3*img_color3


                            perspective_90[p_y, p_x, :] = img_color

                    scipy.misc.imsave(sv3D.panoMeta['panoId'] + '_' + str(degree) + '.png', perspective_90)
                    #scipy.misc.imshow(perspective_90)
                    perspective_90_set.append(perspective_90)
                    #break

                #perspective_90_visual_0 = np.hstack(
                #    (perspective_90_set[3], perspective_90_set[0], perspective_90_set[1], perspective_90_set[2]))
                #perspective_90_visual = np.vstack((ori_pano, perspective_90_visual_0))
                #scipy.misc.imshow(perspective_90_visual)

            if imageSynthesis and i == nearest_pano_idx:
                # Visualize which panorama is the closet
                if needVisual:
                    dataNearest = np.zeros(pano_length + 1, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
                    dataNearest['a_color'] = [0, 1, 0]
                    dataNearest['a_color'][nearest_pano_idx] = [1, 1, 0]
                    dataNearest['a_position'][0:-1] = pano_ori_set
                    dataNearest['a_position'][-1] = offsetPos
                    dataNearest['a_color'][-1] = [1, 0, 0]
                # Calculate synthesis view
                syn_pano = np.zeros((texture_height, texture_width, 3), dtype=np.float32)
                ori_pano = scipy.misc.imresize(sv3D.panorama, (int(texture_height), int(texture_width)), interp='bilinear', mode=None) / 255.0
                #scipy.misc.imshow(ori_pano)
                index_pano = np.full((texture_height, texture_width), False, dtype=bool)
                # indices_split : new indices about planes
                # plane_split : new independent planes {d, nx, ny, nz}
                indices_split = sv3D.indices_split
                plane_split = sv3D.plane_split
                indices_split_reshape = np.reshape(sv3D.indices_split, (256, 512))
                data_all = sv3D.ptCLoudData
                programTexture_set, rrr = [], []
                # For each planes
                for plane_idx in range(0, len(plane_split)):
                    print('{:d}/{:d}'.format(plane_idx, len(plane_split)))
                    # Prepared for texture output
                    if needTexture:
                        # Reset for each plane
                        phase1, phase2, phase3, phase4, crossPhase = False, False, False, False, False
                        texture_xyz_3d, texture_xyz, texture_color, texture_len = [], [], [], 0
                        texture_xyz_3d_back_right, texture_xyz_back_right, texture_color_back_right, texture_len_back_right = [], [], [], 0
                        texture_xyz_3d_back_left, texture_xyz_back_left, texture_color_back_left, texture_len_back_left = [], [], [], 0

                    if planeJoints:
                        aug_plane = np.zeros((256, 512))
                        aug_plane[np.nonzero(indices_split_reshape == plane_idx)] = 1
                        aug_plane = scipy.ndimage.morphology.binary_dilation(aug_plane, iterations=1)
                        aug_plane = np.reshape(aug_plane, (256 * 512))
                        sel = data_all[np.nonzero(aug_plane)]
                    else:
                        sel = data_all[np.nonzero(indices_split == plane_idx)]
                    # For each point
                    for point in sel:
                        if np.isnan(point['a_position']).any():
                            continue
                        x, y, z = point['a_position'] - np.array(offsetPos)
                        if needTexture:
                            if x < 0 and y < 0:
                                phase3 = True
                                if phase4:
                                    crossPhase = True
                            elif x >= 0 and y < 0:
                                phase4 = True
                                if phase3:
                                    crossPhase = True
                        lng, lat = base_process.pos_2_deg(x, y, z)

                        lng = (lng + offsetDeg) % 360
                        img_x = int(lng / 360.0 * texture_width)
                        img_y = int(-(lat - 90) / 180.0 * texture_height)
                        syn_pano[img_y, img_x, :] = point['a_color']
                        index_pano[img_y, img_x] = True
                        if needTexture:
                            if  x >= 0:
                                texture_xyz_3d_back_right.append([x, y, z])
                                texture_xyz_back_right.append([img_x, img_y, 0])
                                texture_color_back_right.append(point['a_color'])
                                texture_len_back_right += 1
                            elif x < 0:
                                texture_xyz_3d_back_left.append([x, y, z])
                                texture_xyz_back_left.append([img_x, img_y, 0])
                                texture_color_back_left.append(point['a_color'])
                                texture_len_back_left += 1

                    #indices_split_visual = np.zeros((256, 512))
                    #indices_split_visual[np.nonzero(indices_split_reshape == plane_idx)] = 1
                    #scipy.misc.imshow(indices_split_visual)
                    #indices_split_visual = np.dstack((indices_split_visual, indices_split_visual, indices_split_visual))
                    #texture_visual = np.hstack((ori_pano, indices_split_visual))
                    #texture_visual_2 = np.hstack((ori_pano*indices_split_visual, syn_pano))
                    #scipy.misc.imshow(np.vstack((texture_visual, texture_visual_2)))

                    if needTexture:
                        if crossPhase:
                            if len(texture_xyz_3d_back_right) >= 3:
                                texture_data = np.zeros((texture_len_back_right),
                                                        dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
                                texture_data['a_color'] = np.array(texture_color_back_right)
                                texture_data['a_position'] = np.array(texture_xyz_back_right)

                                pca = PCA(n_components=2)
                                data_transfer = pca.fit_transform(np.array(texture_xyz_3d_back_right))
                                tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
                                texture_data['a_position'][:, 0] = texture_data['a_position'][:, 0] / 256.0 - 1.0
                                texture_data['a_position'][:, 1] = -(texture_data['a_position'][:, 1] / 128.0 - 1.0)

                                programTexture = glumpy_setting.ProgramTexture(data=texture_data, name='ProgramTexture', tri=tri)
                                programTexture_set.append(programTexture)

                            if len(texture_xyz_3d_back_left) >= 3:
                                texture_data = np.zeros((texture_len_back_left),
                                                        dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
                                texture_data['a_color'] = np.array(texture_color_back_left)
                                texture_data['a_position'] = np.array(texture_xyz_back_left)

                                pca = PCA(n_components=2)
                                data_transfer = pca.fit_transform(np.array(texture_xyz_3d_back_left))
                                tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
                                texture_data['a_position'][:, 0] = texture_data['a_position'][:, 0] / 256.0 - 1.0
                                texture_data['a_position'][:, 1] = -(texture_data['a_position'][:, 1] / 128.0 - 1.0)

                                programTexture = glumpy_setting.ProgramTexture(data=texture_data, name='ProgramTexture', tri=tri)
                                programTexture_set.append(programTexture)
                        else:
                            texture_xyz_3d = texture_xyz_3d_back_right + texture_xyz_3d_back_left
                            texture_xyz = texture_xyz_back_right + texture_xyz_back_left
                            texture_color = texture_color_back_right + texture_color_back_left
                            texture_len = texture_len_back_right + texture_len_back_left
                            if len(texture_xyz_3d) >= 3:
                                texture_data = np.zeros((texture_len),
                                                        dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
                                texture_data['a_color'] = np.array(texture_color)
                                texture_data['a_position'] = np.array(texture_xyz)

                                pca = PCA(n_components=2)
                                data_transfer = pca.fit_transform(np.array(texture_xyz_3d))
                                tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
                                texture_data['a_position'][:, 0] = texture_data['a_position'][:, 0] / 256.0 - 1.0
                                texture_data['a_position'][:, 1] = -(texture_data['a_position'][:, 1] / 128.0 - 1.0)

                                programTexture = glumpy_setting.ProgramTexture(data=texture_data, name='ProgramTexture', tri=tri)
                                programTexture_set.append(programTexture)


                    #break
                print(rrr)
                #scipy.misc.imshow(syn_pano)



        if needVisual:
            programSV3DRegion = glumpy_setting.ProgramSV3DRegion(
                data=data, name='ProgramSV3DRegion', point_size=1,
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
            programSV3DRegionGnd = glumpy_setting.ProgramSV3DRegion(
                data=dataGnd, name='ProgramSV3DRegionGnd', point_size=1,
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
            programSV3DTopology = glumpy_setting.ProgramSV3DTopology(
                data=sv3DRegion.topologyData, name='programSV3DTopology',
                anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d)
            if imageSynthesis:
                programSV3DNearest = glumpy_setting.ProgramSV3DTopology(
                    data=dataNearest, name='programSV3DNearest',
                    anchor_matrix=sv3DRegion.anchorMatrix, anchor_yaw=sv3DRegion.anchorYaw, is_inverse=needMatchInfo3d,
                    point_size=20, alpha=0.6)


"""
For Visualize
"""
if needVisual:
    gpyWindow = glumpy_setting.GpyWindow()
    if createSFM:
        gpyWindow.add_program(programSFM3DRegion)
        gpyWindow.add_program(programTrajectory)

    if createSV:
        if addPlane:
            for j in range(0, pano_length):
                sv3D = sv3DRegion.sv3D_Time[j]
                for i in range(0, len(sv3D.all_plane)):
                    programGround = glumpy_setting.ProgramPlane(data=sv3D.all_plane[i]['data'], name='test',
                                                                face=sv3D.all_plane[i]['tri'])
                    gpyWindow.add_program(programGround)
        else:
            gpyWindow.add_program(programSV3DRegion)
            gpyWindow.add_program(programSV3DTopology)
            if imageSynthesis:
                gpyWindow.add_program(programSV3DNearest)
            if needGround:
                gpyWindow.add_program(programSV3DRegionGnd)

    programAxis = glumpy_setting.ProgramAxis(line_length=5)
    gpyWindow.add_program(programAxis)

    gpyWindow.run()

if needTexture:
    gpyViewWindow = glumpy_setting.GpyViewWindow()
    for programTexture in programTexture_set:
        gpyViewWindow.add_program(programTexture)
    gpyViewWindow.run()