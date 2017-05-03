#!/usr/bin/python3
import numpy as np
import numpy.matlib
import cv2
import scipy.misc
import zlib
import base64
import struct
import os
import json
import skimage.measure
import triangle
from glumpy import glm
from sklearn.decomposition import PCA
#
import base_process
import matplotlib.pyplot as plt

storePath = '/home/andy/src/Google/panometa/'


class StreetView3DRegion:
    def __init__(self, region_id):
        self.regionId = region_id
        self.dataDir = os.path.join(storePath, self.regionId)
        self.metaDir = os.path.join(self.dataDir, "fileMeta.json")
        with open(self.metaDir) as meta_file:
            self.fileMeta = json.load(meta_file)
            meta_file.close()
        """
        # All the google depth maps seem to be store
        # as this size(256, 512), at least what I've seen
        """
        self.sphericalRay = create_spherical_ray(256, 512)
        """
        # For anchor information
        """
        self.anchorId, self.anchorLat, self.anchorLon = None, 0, 0
        self.anchorECEF, self.anchorMatrix, self.anchorYaw = np.zeros(3), np.eye(4, dtype=np.float32), 0
        """
        # Child object StreetView3D
        """
        # without time factor
        self.sv3D_Dict, self.topologyData = {}, None
        # with time factor
        self.panoramaList, self.sv3D_Time, self.sv3D_location = [], [], []
        # TODO query google data><
        self.QQ = False

    def init_region(self, anchor=None):
        """
        Initialize the local region
        Find the anchor
        :return:
        """
        if anchor is None:
            print('Random anchor')
            for panoId in sorted(self.fileMeta['id2GPS']):
                print('anchor is:', panoId, self.fileMeta['id2GPS'][panoId])
                self.anchorId, self.anchorLat, self.anchorLon = \
                    panoId, float(self.fileMeta['id2GPS'][panoId][0]), float(self.fileMeta['id2GPS'][panoId][1])
                self.anchorECEF = base_process.geo_2_ecef(self.anchorLat, self.anchorLon, 22)
                break
        else:
            print('use the anchor')
            print('anchor is:', anchor['anchorId'], anchor['anchorLat'], anchor['anchorLon'])
            self.anchorId, self.anchorLat, self.anchorLon = \
                anchor['anchorId'], anchor['anchorLat'], anchor['anchorLon']
            self.anchorECEF = base_process.geo_2_ecef(self.anchorLat, self.anchorLon, 22)

        # The anchor
        try:
            pano_id_dir = os.path.join(self.dataDir, self.anchorId)
            panorama = scipy.misc.imread(pano_id_dir + '.jpg').astype(np.float)
            with open(pano_id_dir + '.json') as data_file:
                pano_meta = json.load(data_file)
                sv3d = StreetView3D(pano_meta, panorama)
                sv3d.create_ptcloud(self.sphericalRay)
                sv3d.global_adjustment()
                sv3d.local_adjustment(self.anchorECEF)
                # TODO: reduce some redundant work
                #self.sv3D_Dict[self.anchorId] = sv3d
                self.anchorMatrix = np.dot(sv3d.matrix_local, sv3d.matrix_global)
                self.anchorYaw = sv3d.yaw
                data_file.close()

        except FileNotFoundError:
            print('no anchor file QQ')
            self.QQ = True

    def create_region(self):
        for panoId in sorted(self.fileMeta['id2GPS']):
            if panoId == self.anchorId:
                # the anchor
                continue
            pano_id_dir = os.path.join(self.dataDir, panoId)
            panorama = scipy.misc.imread(pano_id_dir + '.jpg').astype(np.float)
            self.panoramaList.append(panorama)
            with open(pano_id_dir + '.json') as data_file:
                pano_meta = json.load(data_file)
                sv3d = StreetView3D(pano_meta, panorama)
                sv3d.create_ptcloud(self.sphericalRay)
                sv3d.global_adjustment()
                sv3d.local_adjustment(self.anchorECEF)
                self.sv3D_Dict[panoId] = sv3d
                self.sv3D_Time.append(sv3d)
                self.sv3D_location.append(sv3d.matrix_local[3, :3])
                data_file.close()

    def create_region_time(self, start=0, end=100):
        pano_t = 0
        pano_set = set()
        for keyframe in sorted(self.fileMeta['keyframe_2_id']):
            panoId = self.fileMeta['keyframe_2_id'][keyframe]
            if panoId in pano_set:
                # already existed
                continue
            pano_set.add(panoId)
            print(keyframe, panoId)
            if end > pano_t >= start:
                pano_id_dir = os.path.join(self.dataDir, panoId)
                panorama = scipy.misc.imread(pano_id_dir + '.jpg').astype(np.float)
                self.panoramaList.append(cv2.imread(pano_id_dir + '.jpg', 0))
                #self.panoramaList.append(scipy.misc.imread(pano_id_dir + '.jpg').astype(np.float32))
                with open(pano_id_dir + '.json') as data_file:
                    pano_meta = json.load(data_file)
                    sv3d = StreetView3D(pano_meta, panorama)
                    sv3d.create_ptcloud(self.sphericalRay)
                    sv3d.global_adjustment()
                    sv3d.local_adjustment(self.anchorECEF)
                    #self.sv3D_Dict[panoId] = sv3d
                    self.sv3D_Time.append(sv3d)
                    self.sv3D_location.append(sv3d.matrix_local[3, :3])
                    data_file.close()
            pano_t += 1

    def create_single(self):
        for panoId in sorted(self.fileMeta['id2GPS']):
            pano_id_dir = os.path.join(self.dataDir, panoId)
            panorama = scipy.misc.imread(pano_id_dir + '.jpg').astype(np.float)
            with open(pano_id_dir + '.json') as data_file:
                pano_meta = json.load(data_file)
                sv3d = StreetView3D(pano_meta, panorama)
                sv3d.create_ptcloud(self.sphericalRay)
                sv3d.global_adjustment()
                sv3d.local_adjustment(self.anchorECEF)
                self.sv3D_Dict[panoId] = sv3d
                data_file.close()
            break

    def create_topoloy(self):
        id_2_gps = self.fileMeta['id2GPS']
        pano_num = len(id_2_gps)
        data = np.zeros(pano_num, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        idx = 0
        for panoid in id_2_gps:
            gps = id_2_gps[panoid]
            ecef = base_process.geo_2_ecef(float(gps[0]), float(gps[1]), 22) - self.anchorECEF
            # TODO: ecef v.s. x-y plane
            data['a_position'][idx] = np.asarray(ecef, dtype=np.float32)
            idx += 1
        data['a_color'] = [0, 1, 0]
        data['a_position'] = base_process.sv3d_apply_m4(data=data['a_position'],
                                   m4=np.linalg.inv(self.anchorMatrix))
        self.topologyData = data

    def create_trajectory(self):
        keyframe_2_id = sorted(self.fileMeta['keyframe_2_id'])
        keyframe_num = len(keyframe_2_id)
        data = np.zeros(pano_num, dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        idx = 0
        for panoid in id_2_gps:
            gps = id_2_gps[panoid]
            ecef = base_process.geo_2_ecef(float(gps[0]), float(gps[1]), 22) - self.anchorECEF
            # TODO: ecef v.s. x-y plane
            data['a_position'][idx] = np.asarray(ecef, dtype=np.float32)
            idx += 1
        data['a_color'] = [0, 1, 0]
        self.trajectoryData = data


class StreetView3D:
    def __init__(self, pano_meta, panorama):
        self.panoMeta, self.panorama = pano_meta, panorama
        self.depthHeader, self.depthMapIndices, self.depthMapPlanes = {}, [], []
        self.depthMap, self.ptCLoudData, self.ptCLoudDataGnd, self.ptCLoudDataGndGrid = None, None, None, None
        self.normal_map, self.gnd_indices = None, None
        self.lat, self.lon, self.yaw = float(pano_meta['Lat']), float(pano_meta['Lon']), float(pano_meta['ProjectionPanoYawDeg'])
        self.ecef = base_process.geo_2_ecef(self.lat, self.lon, 22)

        self.decode_depth_map(pano_meta['rawDepth'])
        if self.depthHeader['panoHeight'] != 256 or self.depthHeader['panoWidth'] != 512:
            print("The depthMap's size of id:%s is unusual: (%d, %d)"
                  % (self.panoMeta['panoId'], self.depthHeader['panoHeight'], self.depthHeader['panoWidth']))

        self.matrix_global = np.eye(4, dtype=np.float32)
        self.matrix_local = np.eye(4, dtype=np.float32)
        self.matrix_offs = np.eye(4, dtype=np.float32)

        self.indices_split, self.plane_split, self.gnd_con, self.non_con, self.all_con = None, None, None, None, None
        self.gnd_plane, self.all_plane = [], []


    def decode_depth_map(self, raw):
        raw = zlib.decompress(base64.urlsafe_b64decode(raw + self.make_padding(raw)))
        pos = 0

        (header_size, num_planes, pano_width, pano_height, plane_indices_offset) = struct.unpack('<BHHHB', raw[0:8])
        self.depthHeader = {'numPlanes': num_planes, 'panoWidth': pano_width, 'panoHeight': pano_height}

        if header_size != 8 or plane_indices_offset != 8:
            print("Invalid depthmap data")
            return
        pos += header_size

        self.depthMapIndices = [x for x in raw[plane_indices_offset:plane_indices_offset + (pano_width * pano_height)]]
        pos += len(self.depthMapIndices)

        self.depthMapPlanes = []
        for i in range(0, num_planes):
            (nx, ny, nz, d) = struct.unpack('<ffff', raw[pos:pos + 16])

            self.depthMapPlanes.append(
                {'d': d, 'nx': nx, 'ny': ny, 'nz': nz})  # nx/ny/nz = unit normal, d = distance from origin
            pos += 16

    @staticmethod
    def make_padding(s):
        return (4 - (len(s) % 4)) * '='

    def create_ptcloud_old_version(self, v):
        height, width = self.depthHeader['panoHeight'], self.depthHeader['panoWidth']
        depth_map = np.zeros([height, width], dtype=np.float32)
        data = np.zeros((height, width), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        panorama = scipy.misc.imresize(self.panorama, (height, width), interp='bilinear', mode=None)
        data['a_color'] = panorama / 255
        for h in range(0, height):
            for w in range(0, width):
                plane_index = self.depthMapIndices[h * width + w]
                plane = self.depthMapPlanes[plane_index]
                if (v[h, w, :].dot(np.array([plane['nx'], plane['ny'], plane['nz']]))) == 0:
                    depth = np.inf
                else:
                    depth = -plane['d'] / v[h, w, :].dot(np.array([plane['nx'], plane['ny'], plane['nz']]))
                depth_map[h][w] = depth
                data[h][w]['a_position'] = depth * v[h, w, :]
        self.depthMap = depth_map
        self.ptCLoudData = data
        self.fix_spherical_inside_out()

    def create_ptcloud(self, v):
        # Create point cloud
        height, width = self.depthHeader['panoHeight'], self.depthHeader['panoWidth']
        plane_indices = np.array(self.depthMapIndices)
        depth_map = np.zeros((height * width), dtype=np.float32)
        gnd_indices = np.zeros((height * width), dtype=np.bool)
        normal_map = np.zeros((height * width, 3), dtype=np.float32)
        v = v.reshape((height * width, 3))

        # index == 0 refers to the sky
        depth_map[np.nonzero(plane_indices == 0)] = np.nan
        # Create depth per plane
        for i in range(0, self.depthHeader['numPlanes']):
            plane = self.depthMapPlanes[i]
            p_depth = np.ones((height * width)) * plane['d']

            vec = (plane['nx'], plane['ny'], plane['nz'])
            normal_map[np.nonzero(plane_indices == i), :] = vec
            angle_diff = base_process.angle_between(vec, (0, 0, -1))
            if angle_diff < 0.3:
                gnd_indices[np.nonzero(plane_indices == i)] = True

            depth = p_depth / v.dot(np.array((plane['nx'], plane['ny'], plane['nz'])))
            #depth = np.ones((height * width)) * -10
            depth_map[np.nonzero(plane_indices == i)] = depth[np.nonzero(plane_indices == i)]

        panorama = scipy.misc.imresize(self.panorama, (height, width), interp='bilinear', mode=None)
        xyz = (np.transpose(v) * np.matlib.repmat(depth_map, 3, 1))
        data = np.zeros((height*width), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        data['a_position'] = np.transpose(xyz)
        data['a_color'] = np.reshape(np.array(panorama) / 255, (height*width, 3))

        # Filter out nan point
        # Split ground with non-ground
        con = ~np.isnan(data['a_position'][:, 0])
        con &= ~np.isnan(data['a_position'][:, 1])
        con &= ~np.isnan(data['a_position'][:, 2])
        non_con = con & ~gnd_indices
        gnd_con = con & gnd_indices

        # Store
        self.depthMap = depth_map.reshape((height, width))
        self.gnd_indices = gnd_indices.reshape((height, width))
        self.normal_map = normal_map.reshape(height, width, 3)
        self.ptCLoudData = data
        self.gnd_con = gnd_con
        self.non_con = non_con
        self.all_con = con
        # This will split planes further
        # indices_split : new indices about planes
        # plane_split : new independent planes {d, nx, ny, nz}
        self.split_plane()
        #self.fix_spherical_inside_out()
        #self.visualize()
        #self.auto_plane()

    def auto_plane_gnd(self):
        # TODO the condition is no need
        indices_split_gnd = self.indices_split[np.nonzero(self.gnd_con)]
        data_gnd = self.ptCLoudData[np.nonzero(self.gnd_con)]
        plane_split = self.plane_split

        for i in range(1, len(plane_split)):
            plane = plane_split[i]
            vec = (plane['nx'], plane['ny'], plane['nz'])
            angle_diff = base_process.angle_between(vec, (0, 0, -1))
            if angle_diff < 0.3 or True:
                sel = data_gnd[np.nonzero(indices_split_gnd == i)]
                if len(sel) < 3:
                    continue
                pca = PCA(n_components=2)
                data_transfer = pca.fit_transform(sel['a_position'])
                tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
                self.gnd_plane.append({'data': sel, 'tri': tri})

    def auto_plane(self):
        indices_split = self.indices_split[np.nonzero(self.all_con)]
        data_gnd = self.ptCLoudData[np.nonzero(self.all_con)]
        plane_split = self.plane_split

        for i in range(1, len(plane_split)):
            sel = data_gnd[np.nonzero(indices_split == i)]
            if len(sel) < 3:
                continue
            pca = PCA(n_components=2)
            data_transfer = pca.fit_transform(sel['a_position'])
            tri = np.array(triangle.delaunay(data_transfer), dtype=np.uint32)
            self.all_plane.append({'data': sel, 'tri': tri})

    def visualize(self):
        self.show_pano()
        self.show_index()
        self.show_normal()
        self.show_depth()
        self.show_index_split()
        self.show_gnd()

    def split_plane(self):
        height, width = self.depthHeader['panoHeight'], self.depthHeader['panoWidth']
        indices = np.array(self.depthMapIndices).reshape(height, width)
        indices_split = np.zeros((height, width), dtype=np.float32)
        cur_idx = 0
        plane_split = []
        for i in range(0, self.depthHeader['numPlanes']):
            index_map = np.zeros((height, width), dtype=np.float32)
            index_map[np.nonzero(indices == i)] = 1
            plane = self.depthMapPlanes[i]
            all_labels = skimage.measure.label(index_map, background=0, connectivity=1)
            for l in range(1, all_labels.max()+1):
                label_map = np.zeros((height, width), dtype=np.float32)
                label_map[np.nonzero(all_labels == l)] = 1
                indices_split[np.nonzero(all_labels == l)] = cur_idx
                plane_split.append(plane)
                cur_idx += 1

        self.indices_split = indices_split.reshape((height * width))
        self.plane_split = plane_split

    def create_ptcloud_ground_grid(self):
        data = base_process.create_plane(n_point=80, sx=5, sy=8, ground_z=-2)
        data = data[np.nonzero(~np.isnan(data['a_position'][:, :, 0]))]

        panorama = self.panorama
        pano_height, pano_width = panorama.shape[0], panorama.shape[1]
        for idx in range(len(data['a_position'])):
            vec = base_process.unit_vector(data['a_position'][idx, :])
            lat, lng = base_process.vec_2_panorama(vec)

            color_canvas_x = int(((lng + 180) / 360) * pano_width)
            color_canvas_y = int((-(lat - 90) / 180) * pano_height)

            color = panorama[color_canvas_y, color_canvas_x, :] / 255
            data['a_color'][idx, :] = color

        data['a_position'][:, 1] = -data['a_position'][:, 1]
        self.ptCLoudDataGndGrid = data

    def fix_spherical_inside_out(self):
        """
        panorama's four corner v.s center point
        if the spherical_ray is center point, then this process need to be done
        I reverse the y-axis make it inside-out
        :return:fixed  ptCLoudData
        """
        self.ptCLoudData['a_position'][:, 1] = -self.ptCLoudData['a_position'][:, 1]
        self.ptCLoudDataGnd['a_position'][:, 1] = -self.ptCLoudDataGnd['a_position'][:, 1]
        m4 = np.array([[ -1.00000000e+00,   0.00000000e+00,  -1.22464685e-16,   0.00000000e+00],
              [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,   0.00000000e+00],
              [  1.22464685e-16,   0.00000000e+00,  -1.00000000e+00,   0.00000000e+00],
              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

        self.ptCLoudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudData['a_position'],
                                                                    m4=m4)
        self.ptCLoudDataGnd['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudDataGnd['a_position'], m4=m4)

    def global_adjustment(self):
        matrix = np.eye(4, dtype=np.float32)
        # Change xy-plan to ecef coordinate
        glm.rotate(matrix, 90, 0, 1, 0)
        glm.rotate(matrix, 90, 1, 0, 0)
        glm.rotate(matrix, self.yaw, -1, 0, 0)
        glm.rotate(matrix, self.lat, 0, -1, 0)
        glm.rotate(matrix, self.lon, 0, 0, 1)
        self.matrix_global = matrix

    def local_adjustment(self, anchor_ecef):
        matrix = np.eye(4, dtype=np.float32)
        ecef = self.ecef - anchor_ecef
        glm.translate(matrix, ecef[0], ecef[1], ecef[2])
        self.matrix_local = matrix

    def apply_global_adjustment(self):
        self.ptCLoudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudData['a_position'], m4=self.matrix_global)

    def apply_local_adjustment(self):
        self.ptCLoudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudData['a_position'], m4=self.matrix_local)

    def apply_anchor_adjustment(self, anchor_matrix):
        self.ptCLoudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudData['a_position'],
                                                                    m4=np.linalg.inv(anchor_matrix))
    def show_pano(self):
        panorama = self.panorama
        #panorama[100-10:100+10, 100-10:100+10, :] = [255, 0, 0]
        scipy.misc.imshow(panorama)

    def show_index(self):
        height = self.depthHeader['panoHeight']
        width = self.depthHeader['panoWidth']
        indices = np.array(self.depthMapIndices)
        index_map_all = np.zeros((height * width, 3), dtype=np.uint8)
        color_map = np.random.random_integers(0, 255, (self.depthHeader['numPlanes'], 3))
        for i in range(0, self.depthHeader['numPlanes']):
            index_map = np.zeros((height * width, 3), dtype=np.uint8)
            index_map[np.nonzero(indices == i), :] = color_map[i, :]
            index_map_all[np.nonzero(indices == i), :] = color_map[i, :]
            #cv2.imwrite('index{}.png'.format(str(i)), index_map.reshape((height, width, 3)))

        #cv2.imwrite('index_map.png', index_map_all.reshape((height, width, 3)))
        #cv2.imshow('image', index_map_all.reshape((height, width, 3)).astype(np.uint8))
        #cv2.waitKey(0)
        scipy.misc.imshow(index_map_all.reshape((height, width, 3)).astype(np.uint8))

    def show_normal(self):
        scipy.misc.imshow(self.normal_map)

    def show_depth(self):
        # The further, the brighter
        # Inverse to inside-out
        depth_map = -self.depthMap * 255 / 50
        depth_map[np.nonzero(np.isnan(depth_map))] = 255
        depth_map[np.nonzero(depth_map > 255)] = 255
        depth_map /= 255
        scipy.misc.imshow(depth_map)
        scipy.misc.imsave('depth.png', depth_map)

    def show_index_split(self):
        height = self.depthHeader['panoHeight']
        width = self.depthHeader['panoWidth']
        indices = np.array(self.indices_split)
        length = len(self.plane_split)
        index_map_all = np.zeros((height * width, 3), dtype=np.uint8)
        color_map = np.random.random_integers(0, 255, (length, 3))
        for i in range(0, length):
            index_map = np.zeros((height * width, 3), dtype=np.uint8)
            index_map[np.nonzero(indices == i), :] = color_map[i, :]
            index_map_all[np.nonzero(indices == i), :] = color_map[i, :]
            #cv2.imwrite('index{}.png'.format(str(i)), index_map.reshape((height, width, 3)))

        #cv2.imwrite('index_map.png', index_map_all.reshape((height, width, 3)))
        #cv2.imshow('image', index_map_all.reshape((height, width, 3)).astype(np.uint8))
        #cv2.waitKey(0)
        scipy.misc.imshow(index_map_all.reshape((height, width, 3)).astype(np.uint8))

    def show_gnd(self):
        scipy.misc.imshow(self.gnd_indices)

def create_spherical_ray(height, width):
    h = np.arange(height)
    theta = (height - h - 0.5) / height * np.pi
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    w = np.arange(width)
    phi = (width - w - 0.5) / width * 2 * np.pi + np.pi / 2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    v = np.zeros((height, width, 3))
    ## interesting
    v[:, :, 0] = sin_theta.reshape((height, 1)) * cos_phi
    v[:, :, 1] = sin_theta.reshape((height, 1)) * sin_phi
    v[:, :, 2] = cos_theta.reshape((height, 1)) * np.ones(width)

    return v


class StreetView3DRegionnnn:
    def __init__(self, fileID):
        fname = '/home/andy/src/Google/panometa/' + fileID + '/fileMeta.json'
        if os.path.isfile(fname):
            print('Successfully find the existing region"' + fileID + '"(accroding to the fileMeta):')
            with open(fname) as data_file:
                fileMeta = json.load(data_file)
                self.panoDict = fileMeta['id2GPS']
                data_file.close()
        else:
            print('Fail to open the file or path doesn\'t exit')

    def createTopoloy(self):
        id2GPS = self.panoDict
        panoNum = len(id2GPS)
        data = np.zeros((panoNum), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        ECEF = []
        for id in id2GPS:
            GPS = id2GPS[id]
            ECEF.append(base_process.geo2ECEF(float(GPS[0]), float(GPS[1])))
        data['a_color'] = [1, 1, 1]
        data['a_position'] = np.asarray(ECEF, dtype=np.float32)
        # data['a_position'] -= data[0]['a_position']
        self.topology = data
        return data