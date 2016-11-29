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
from glumpy import glm

#
import base_process

storePath = '/home/andy/src/Google/panometa/'


class StreetView3DRegion:
    def __init__(self, region_id):
        self.regionId = region_id
        self.dataDir = os.path.join(storePath, self.regionId)
        self.metaDir = os.path.join(self.dataDir, "fileMeta.json")
        with open(self.metaDir) as meta_file:
            self.fileMeta = json.load(meta_file)
            meta_file.close()

        # All the google depth maps seem to be store
        # as this size(256, 512), at least what I've seen
        self.sphericalRay = create_spherical_ray(256, 512)

        self.anchorId, self.anchorLat, self.anchorLon = None, 0, 0
        self.anchorECEF, self.anchorMatrix, self.anchorYaw = np.zeros(3), np.eye(4, dtype=np.float32), 0
        self.sv3D_Dict, self.topologyData = {}, None

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
                self.sv3D_Dict[self.anchorId] = sv3d
                self.anchorMatrix = np.dot(sv3d.matrix_local, sv3d.matrix_global)
                self.anchorYaw = sv3d.yaw
                data_file.close()

        except:
            print('no anchor file QQ')
            self.QQ = True

    def create_region(self):
        for panoId in self.fileMeta['id2GPS']:
            if panoId == self.anchorId:
                continue
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

    def create_single(self):
        for panoId in self.fileMeta['id2GPS']:
            if panoId == self.anchorId:
                continue
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
            data['a_position'][idx] = np.asarray([ecef[1], ecef[2], ecef[0]], dtype=np.float32)
            idx += 1
        data['a_color'] = [0, 1, 0]
        self.topologyData = data

class StreetView3D:
    def __init__(self, pano_meta, panorama):
        self.panoMeta = pano_meta
        self.panorama = panorama
        self.depthHeader, self.depthMapIndices, self.depthMapPlanes = {}, [], []
        self.depthMap, self.ptCLoudData, self.ptCLoudDataGnd, self.ptCLoudDataGndGrid = None, None, None, None
        self.lat, self.lon, self.yaw = float(pano_meta['Lat']), float(pano_meta['Lon']), float(pano_meta['ProjectionPanoYawDeg'])
        self.ecef = base_process.geo_2_ecef(self.lat, self.lon, 22)

        self.decode_depth_map(pano_meta['rawDepth'])
        if self.depthHeader['panoHeight'] != 256 or self.depthHeader['panoWidth'] != 512:
            print("The depthMap's size of id:%s is unusual: (%d, %d)"
                  % (self.panoMeta['panoId'], self.depthHeader['panoHeight'], self.depthHeader['panoWidth']))

        self.matrix_global = np.eye(4, dtype=np.float32)
        self.matrix_local = np.eye(4, dtype=np.float32)
        self.matrix_offs = np.eye(4, dtype=np.float32)


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
        height, width = self.depthHeader['panoHeight'], self.depthHeader['panoWidth']
        plane_indices = np.array(self.depthMapIndices)
        depth_map = np.zeros((height * width), dtype=np.float32)
        depth_map[np.nonzero(plane_indices == 0)] = np.nan
        depth_map_gnd = np.copy(depth_map)
        v = v.reshape((height * width, 3))

        # index == 0 refers to the sky
        for i in range(1, self.depthHeader['numPlanes']):
            plane = self.depthMapPlanes[i]
            p_depth = np.ones((height * width)) * plane['d']

            vec = (plane['nx'], plane['ny'], plane['nz'])
            angle_diff = base_process.angle_between(vec, (0, 0, -1))
            if angle_diff < 0.3:
                depth = np.ones((height * width)) * np.nan
                depth_gnd = -p_depth / v.dot(np.array((plane['nx'], plane['ny'], plane['nz'])))
            else:
                depth = -p_depth / v.dot(np.array((plane['nx'], plane['ny'], plane['nz'])))
                depth_gnd = np.ones((height * width)) * np.nan

            #depth = -p_depth / v.dot(np.array((plane['nx'], plane['ny'], plane['nz'])))

            depth_map[np.nonzero(plane_indices == i)] = depth[np.nonzero(plane_indices == i)]
            depth_map_gnd[np.nonzero(plane_indices == i)] = depth_gnd[np.nonzero(plane_indices == i)]

        panorama = scipy.misc.imresize(self.panorama, (height, width), interp='bilinear', mode=None)
        xyz = (np.transpose(v) * np.matlib.repmat(depth_map, 3, 1))
        data = np.zeros((height, width), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        data['a_position'] = np.transpose(xyz).reshape((height, width, 3))
        data['a_color'] = np.array(panorama) / 255

        con = ~np.isnan(data['a_position'][:, :, 0])
        con &= ~np.isnan(data['a_position'][:, :, 1])
        con &= ~np.isnan(data['a_position'][:, :, 2])
        data = data[np.nonzero(con)]
        #data = data.flatten()
        """
        GROUND PLANE!
        """
        xyz = (np.transpose(v) * np.matlib.repmat(depth_map_gnd, 3, 1))
        data_gnd = np.zeros((height, width), dtype=[('a_position', np.float32, 3), ('a_color', np.float32, 3)])
        data_gnd['a_position'] = np.transpose(xyz).reshape((height, width, 3))
        data_gnd['a_color'] = np.array(panorama) / 255

        con = ~np.isnan(data_gnd['a_position'][:, :, 0])
        con &= ~np.isnan(data_gnd['a_position'][:, :, 1])
        con &= ~np.isnan(data_gnd['a_position'][:, :, 2])

        #con &= (data_gnd['a_position'][:, :, 0] < 10)
        #con &= (data_gnd['a_position'][:, :, 0] > -10)
        #con &= (data_gnd['a_position'][:, :, 1] < 10)
        #con &= (data_gnd['a_position'][:, :, 1] > -10)
        # con &= (data_gnd['a_position'][:, :, 2] < 10)
        # con &= (data_gnd['a_position'][:, :, 2] > -10)
        data_gnd = data_gnd[np.nonzero(con)]

        self.depthMap = depth_map.reshape((height, width))
        self.ptCLoudData = data
        self.ptCLoudDataGnd = data_gnd
        self.fix_spherical_inside_out()


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

    def show_depth(self):
        # The further, the brighter
        # Inverse to inside-out
        depth_map = self.depthMap * 255 / 50
        depth_map[np.nonzero(np.isnan(depth_map))] = 255
        depth_map[np.nonzero(depth_map > 255)] = 255
        depth_map /= 255
        #scipy.misc.imshow(depth_map)
        #scipy.misc.imshow(self.panorama)
        scipy.misc.imsave('depth.png', depth_map)

    def show_pano(self):
        panorama = self.panorama
        #panorama[100-10:100+10, 100-10:100+10, :] = [255, 0, 0]

        scipy.misc.imshow(panorama)

    def global_adjustment(self):
        matrix = np.eye(4, dtype=np.float32)
        #glm.rotate(matrix, 180, 0, 1, 0)
        glm.rotate(matrix, self.yaw, 0, 0, -1)
        glm.rotate(matrix, self.lat, -1, 0, 0)
        glm.rotate(matrix, self.lon, 0, 1, 0)
        self.matrix_global = matrix

    def local_adjustment(self, anchor_ecef):
        matrix = np.eye(4, dtype=np.float32)
        ecef = self.ecef - anchor_ecef
        glm.translate(matrix, ecef[1], ecef[2], ecef[0])
        self.matrix_local = matrix

    def apply_global_adjustment(self):
        self.ptCLoudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudData['a_position'],
                                                                    m4=self.matrix_global)

        self.ptCLoudDataGnd['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudDataGnd['a_position'],
                                                                       m4=self.matrix_global)

#        self.ptCLoudDataGndGrid['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudDataGndGrid['a_position'],
#                                                                       m4=self.matrix_global)

    def apply_local_adjustment(self):
        self.ptCLoudData['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudData['a_position'],
                                                                    m4=self.matrix_local)

        self.ptCLoudDataGnd['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudDataGnd['a_position'],
                                                                       m4=self.matrix_local)

#        self.ptCLoudDataGndGrid['a_position'] = base_process.sv3d_apply_m4(data=self.ptCLoudDataGndGrid['a_position'],
#                                                                       m4=self.matrix_local)


    def showIndex(self):
        height = self.DepthHeader['panoHeight']
        width = self.DepthHeader['panoWidth']
        indices = np.array((self.DepthMapIndices))
        indexMap = np.zeros((height * width, 3), dtype=np.uint8)
        colorMap = np.random.random_integers(0, 255, (self.DepthHeader['numPlanes'], 3))
        for i in range(0, self.DepthHeader['numPlanes']):
            indexMap = np.zeros((height * width, 3), dtype=np.uint8)
            indexMap[np.nonzero(indices == i), :] = colorMap[i, :]
            cv2.imwrite('index' + str(i) + '.png', indexMap.reshape((height, width, 3)))

        # indexMap *=  255 / 50
        # indexMap[np.nonzero(indexMap > 255)] = 255
        cv2.imshow('image', indexMap.reshape((height, width, 3)).astype(np.uint8))
        cv2.waitKey(0)


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