#!/usr/bin/python2
import numpy as np
import cv2
import urllib
from urllib.request import urlopen
import urllib.request
from lxml import etree
import lxml
from PIL import Image
from io import BytesIO
import requests
import json
import os


"""
Object can fetch google's data
also deal with different pathPoints in various form
"""
storePath = '/home/andy/src/Google/panometa/'


class PanoFetcher:
    def __init__(self, zoom=1, radius=30):
        self.zoom = zoom  # Get what pano
        self.radius = radius  # Panorama's' size

        self.panoSet, self.panoList, self.panoDict, self.cur = set(), [], {}, 0

    @staticmethod
    def file_dir_exist(file_id):
        if not os.path.exists(storePath + file_id):
            os.makedirs(storePath + file_id)
            return False
        return True

    @staticmethod
    def store_pano(store_dir, pano_basic, img):
        with open(store_dir + '.json', 'w') as outfile:
            json.dump(pano_basic, outfile)
            outfile.close()
        cv2.imwrite(store_dir + '.jpg', img)

    def store_bfs_meta(self, file_id):
        with open(storePath + file_id + '/fileMeta.json', 'w') as outfile:
            file_meta = {'panoList': self.panoList, 'cur': self.cur, 'id2GPS': self.panoDict}
            print("id2GPS's length: %d" % len(self.panoDict))
            print("panoList's length: %d" % len(self.panoList))
            json.dump(file_meta, outfile)
            outfile.close()

    def get_new_pano_meta(self):
        img, pano_basic, pano = None, None, None
        try:  # THIS is so serious
            pano = get_panorama_metadata(panoid=self.panoList[self.cur], radius=self.radius)
            self.panoDict[pano.PanoId] = [pano.Lat, pano.Lon]
            img = get_panorama(pano.PanoId, zoom=self.zoom)
            try:
                pano_basic = {'panoId': pano.PanoId, 'Lat': pano.Lat,
                              'Lon': pano.Lon, 'ProjectionPanoYawDeg': pano.ProjectionPanoYawDeg,
                              'AnnotationLinks': pano.AnnotationLinks, 'rawDepth': pano.rawDepth,
                              'Text': pano.Text}
                # Add new founded panos into the list
                for link in pano.AnnotationLinks:
                    if (link['PanoId'] not in self.panoSet):
                        pano_id = link['PanoId']
                        self.panoList.append(pano_id)
                        self.panoSet.add(pano_id)
            except:
                print(pano.PanoId + ' lacks some important data.')
                pano_basic = {}
        except:
            print("Google what's worng with " + self.panoList[self.cur])
        return img, pano_basic, pano.PanoId

    def bfs(self, file_id, start_gps, max_pano=100):
        self.file_dir_exist(file_id)
        # Initialize panoList, panoDict, panoSet, cur
        self.panoSet, self.panoList, self.panoDict, self.cur = set(), [], {}, 0
        # Change the GPS to panoID
        # and put first panoID into the list
        (lat, lon) = start_gps
        pano = get_panorama_metadata(lat=lat, lon=lon, radius=self.radius)
        self.panoSet.add(pano.PanoId)
        self.panoList.append(pano.PanoId)
        # Until maximum
        for self.cur in range(0, max_pano):
            # Get the pano according to the list
            img, pano_basic, pano_id = self.get_new_pano_meta()
            store_dir = storePath + file_id + '/' + pano_id
            self.store_pano(store_dir, pano_basic, img)
            print(self.cur, pano_id)
        # Store
        self.store_bfs_meta(file_id)

    def bfs_aug(self, file_id, start_gps=None, max_pano=100):
        file_name = storePath + file_id + '/fileMeta.json'
        if os.path.isfile(file_name):
            print('Augment the existing region"' + file_id + '"(according to the fileMeta):')
            # Initialize panoList, panoDict, panoSet, cur
            with open(file_name) as data_file:
                file_meta = json.load(data_file)
                self.panoList = file_meta['panoList']
                cur = file_meta['cur'] + 1
                self.panoDict = file_meta['id2GPS']
                print(self.panoList, cur - 1)
                self.panoSet = set(self.panoList)
                self.cur = cur
                data_file.close()
            # Until maximum
            for self.cur in range(cur + 0, cur + max_pano):
                # Get the pano accroding to the list
                img, pano_basic, pano_id = self.get_new_pano_meta()
                store_dir = storePath + file_id + '/' + pano_id
                self.store_pano(store_dir, pano_basic, img)
                print(self.cur, pano_id)
            # Store
            self.store_bfs_meta(file_id)
        else:
            print('Create the region "' + file_id + '" first time.(Or lacking the fileMeta):')
            self.bfs(file_id, start_gps, max_pano)
            return

    def info_3d(self, file_id, path_point_set_info3d):
        pano_set, id_2_gps, info_2_id = set(), {}, {}
        file_id += '_info3d'
        if self.file_dir_exist(file_id):
            print('This info_3d file already exists.')
        else:
            print('Create new info_3d panometa.')
            for pathPoint in path_point_set_info3d:
                [lat, lon] = pathPoint.split(',')
                pano = get_panorama_metadata(lat=lat, lon=lon, radius=self.radius)
                print(lat, lon, pano.PanoId)
                info_2_id[pathPoint] = pano.PanoId
                id_2_gps[pano.PanoId] = [pano.Lat, pano.Lon]
                if pano.PanoId not in pano_set:
                    pano_set.add(pano.PanoId)
                    img = get_panorama(pano.PanoId, zoom=self.zoom)
                    try:
                        pano_basic = {'panoId': pano.PanoId, 'Lat': pano.Lat,
                                      'Lon': pano.Lon, 'ProjectionPanoYawDeg': pano.ProjectionPanoYawDeg,
                                      'AnnotationLinks': pano.AnnotationLinks, 'rawDepth': pano.rawDepth,
                                      'Text': pano.Text}
                    except:
                        print(pano.PanoId + ' lacks some important data.')
                        pano_basic = {}
                    store_dir = storePath + file_id + '/' + pano.PanoId
                    self.store_pano(store_dir, pano_basic, img)
            with open(storePath + file_id + '/fileMeta.json', 'w') as outfile:
                file_meta = {'info2ID': info_2_id, 'id2GPS': id_2_gps}
                print("id2GPS's length: %d" % len(id_2_gps))
                json.dump(file_meta, outfile)
                outfile.close()

    def info_bfs(self, file_id):
        file_id += '_info3d'
        file_name = storePath + file_id + '/fileMeta.json'
        if os.path.isfile(file_name):
            print('Augment the existing region"' + file_id + '"(according to the fileMeta):')
            # Initialize panoList, panoDict, panoSet, cur
            with open(file_name) as data_file:
                file_meta = json.load(data_file)
                self.panoList = file_meta['panoList']
                cur = file_meta['cur'] + 1
                self.panoDict = file_meta['id2GPS']
                print(self.panoList, cur - 1)
                self.panoSet = set(self.panoList)
                self.cur = cur
                data_file.close()
            # Until maximum
            for self.cur in range(cur + 0, cur + max_pano):
                # Get the pano accroding to the list
                img, pano_basic, pano_id = self.get_new_pano_meta()
                store_dir = storePath + file_id + '/' + pano_id
                self.store_pano(store_dir, pano_basic, img)
                print(self.cur, pano_id)
            # Store
            self.store_bfs_meta(file_id)
        else:
            print('The region "' + file_id + '" has not created yet.')
            return

    def trajectory(self, file_id, path_trajectory):
        pano_set, id_2_gps, keyframe_2_id = set(), {}, {}
        file_id += '_trajectory'
        if self.file_dir_exist(file_id):
            print('This trajectory file already exists.')
        else:
            print('Create new trajectory panometa.')
            for keyframe, gps in sorted(path_trajectory.items()):
                lat, lon, h = gps
                pano = get_panorama_metadata(lat=lat, lon=lon, radius=self.radius)
                print(keyframe, lat, lon, pano.PanoId)
                id_2_gps[pano.PanoId] = [pano.Lat, pano.Lon]
                keyframe_2_id[keyframe] = pano.PanoId
                if pano.PanoId not in pano_set:
                    pano_set.add(pano.PanoId)
                    img = get_panorama(pano.PanoId, zoom=self.zoom)
                    try:
                        pano_basic = {'panoId': pano.PanoId, 'Lat': pano.Lat,
                                      'Lon': pano.Lon, 'ProjectionPanoYawDeg': pano.ProjectionPanoYawDeg,
                                      'AnnotationLinks': pano.AnnotationLinks, 'rawDepth': pano.rawDepth,
                                      'Text': pano.Text}
                    except:
                        print(pano.PanoId + ' lacks some important data.')
                        pano_basic = {}
                    store_dir = storePath + file_id + '/' + pano.PanoId
                    self.store_pano(store_dir, pano_basic, img)
            with open(storePath + file_id + '/fileMeta.json', 'w') as outfile:
                file_meta = {'id2GPS': id_2_gps, 'keyframe_2_id': keyframe_2_id}
                print("id2GPS's length: %d" % len(id_2_gps))
                json.dump(file_meta, outfile)
                outfile.close()


# Something I don't familiar
# This object helps me a alot to parse the google data					
class PanoramaMetadata:
    def __init__(self, panodoc):
        try:
            self.PanoDoc = panodoc
            panoDocCtx = self.PanoDoc.xpathNewContext()

            self.PanoId = panoDocCtx.xpathEval("/panorama/data_properties/@pano_id")[0].content
            # self.ImageWidth = panoDocCtx.xpathEval("/panorama/data_properties/@image_width")[0].content
            # self.ImageHeight = panoDocCtx.xpathEval("/panorama/data_properties/@image_height")[0].content
            # self.TileWidth = panoDocCtx.xpathEval("/panorama/data_properties/@tile_width")[0].content
            # self.TileHeight = panoDocCtx.xpathEval("/panorama/data_properties/@tile_height")[0].content
            # self.NumZoomLevels = panoDocCtx.xpathEval("/panorama/data_properties/@num_zoom_levels")[0].content
            self.Lat = panoDocCtx.xpathEval("/panorama/data_properties/@lat")[0].content
            self.Lon = panoDocCtx.xpathEval("/panorama/data_properties/@lng")[0].content
            # self.OriginalLat = panoDocCtx.xpathEval("/panorama/data_properties/@original_lat")[0].content
            # self.OriginalLon = panoDocCtx.xpathEval("/panorama/data_properties/@original_lng")[0].content
            # self.Copyright = panoDocCtx.xpathEval("/panorama/data_properties/copyright/text()")[0].content
            # some panorama hasn't the files follow
            # which will cause error
            try:
                self.Text = panoDocCtx.xpathEval("/panorama/data_properties/text/text()")[0].content
            except:
                self.Text = ''
            # self.Region = panoDocCtx.xpathEval("/panorama/data_properties/region/text()")[0].content
            # self.Country = panoDocCtx.xpathEval("/panorama/data_properties/country/text()")[0].content

            # self.ProjectionType = panoDocCtx.xpathEval("/panorama/projection_properties/@projection_type")[0].content
            self.ProjectionPanoYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@pano_yaw_deg")[0].content
            # self.ProjectionTiltYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_yaw_deg")[0].content
            # self.ProjectionTiltPitchDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_pitch_deg")[0].content

            self.AnnotationLinks = []
            for cur in panoDocCtx.xpathEval("/panorama/annotation_properties/link"):
                self.AnnotationLinks.append({'YawDeg': cur.xpathEval("@yaw_deg")[0].content,
                                             'PanoId': cur.xpathEval("@pano_id")[0].content,
                                             'RoadARGB': cur.xpathEval("@road_argb")[0].content
                                             # ,'Text': text
                                             # some panorama hasn't this file
                                             # which will cause error
                                             # text = cur.xpathEval("link_text/text()")[0].content
                                             })

            tmp = panoDocCtx.xpathEval("/panorama/model/depth_map/text()")
            if len(tmp) > 0:
                self.rawDepth = tmp[0].content
        except:
            print(self.PanoId + 'has some problem.')


class PanoramaMetadata_python3:
    def __init__(self, root):
        try:
            #self.PanoDoc = panodoc
            #panoDocCtx = self.PanoDoc.xpathNewContext()

            #self.PanoId = panoDocCtx.xpathEval("/panorama/data_properties/@pano_id")[0].content
            self.PanoId = root.find('data_properties').get('pano_id')

            # self.ImageWidth = panoDocCtx.xpathEval("/panorama/data_properties/@image_width")[0].content
            # self.ImageHeight = panoDocCtx.xpathEval("/panorama/data_properties/@image_height")[0].content
            # self.TileWidth = panoDocCtx.xpathEval("/panorama/data_properties/@tile_width")[0].content
            # self.TileHeight = panoDocCtx.xpathEval("/panorama/data_properties/@tile_height")[0].content
            # self.NumZoomLevels = panoDocCtx.xpathEval("/panorama/data_properties/@num_zoom_levels")[0].content
            #self.Lat = panoDocCtx.xpathEval("/panorama/data_properties/@lat")[0].content
            #self.Lon = panoDocCtx.xpathEval("/panorama/data_properties/@lng")[0].content
            self.Lat = root.find('data_properties').get('lat')
            self.Lon = root.find('data_properties').get('lng')
            # self.OriginalLat = panoDocCtx.xpathEval("/panorama/data_properties/@original_lat")[0].content
            # self.OriginalLon = panoDocCtx.xpathEval("/panorama/data_properties/@original_lng")[0].content
            # self.Copyright = panoDocCtx.xpathEval("/panorama/data_properties/copyright/text()")[0].content
            # some panorama hasn't the files follow
            # which will cause error
            try:
                #self.Text = panoDocCtx.xpathEval("/panorama/data_properties/text/text()")[0].content
                self.Text = root.find('data_properties').find('text').text
            except:
                self.Text = ''
            # self.Region = panoDocCtx.xpathEval("/panorama/data_properties/region/text()")[0].content
            # self.Country = panoDocCtx.xpathEval("/panorama/data_properties/country/text()")[0].content

            # self.ProjectionType = panoDocCtx.xpathEval("/panorama/projection_properties/@projection_type")[0].content
            #self.ProjectionPanoYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@pano_yaw_deg")[0].content
            self.ProjectionPanoYawDeg = root.find('projection_properties').get('pano_yaw_deg')
            # self.ProjectionTiltYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_yaw_deg")[0].content
            # self.ProjectionTiltPitchDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_pitch_deg")[0].content

            self.AnnotationLinks = []
            for cur in root.find('annotation_properties').iterfind('link'):
                self.AnnotationLinks.append({'YawDeg': cur.get('yaw_deg'),
                                             'PanoId': cur.get('pano_id'),
                                             'RoadARGB': cur.get('road_argb')
                                             # ,'Text': text
                                             # some panorama hasn't this file
                                             # which will cause error
                                             # text = cur.xpathEval("link_text/text()")[0].content
                                             })
            '''
            for cur in panoDocCtx.xpathEval("/panorama/annotation_properties/link"):
                self.AnnotationLinks.append({'YawDeg': cur.xpathEval("@yaw_deg")[0].content,
                                             'PanoId': cur.xpathEval("@pano_id")[0].content,
                                             'RoadARGB': cur.xpathEval("@road_argb")[0].content
                                             # ,'Text': text
                                             # some panorama hasn't this file
                                             # which will cause error
                                             # text = cur.xpathEval("link_text/text()")[0].content
                                             })
            '''

            tmp = root.find('model').find('depth_map').text
            self.rawDepth = root.find('model').find('depth_map').text

        except:
            print(self.PanoId + 'has some problem.')


def get_url_contents(url):
    f = urlopen(url)
    data = f.read()
    f.close()
    return data


# panoid is the value from panorama metadata
# OR: supply lat/lon/radius to find the nearest pano to lat/lon within radius
def get_panorama_metadata(panoid=None, lat=None, lon=None, radius=30):
    base_uri = 'http://maps.google.com/cbk'
    url = '%s?'
    url += 'output=xml'  # metadata output
    url += '&v=4'  # version
    url += '&dm=1'  # include depth map
    url += '&pm=1'  # include pano map
    if panoid is None:
        url += '&ll=%s,%s'  # lat/lon to search at
        url += '&radius=%s'  # search radius
        url = url % (base_uri, lat, lon, radius)
    else:
        url += '&panoid=%s'  # panoid to retrieve
        url = url % (base_uri, panoid)

    request = urllib.request.Request(url)
    xml_text = urllib.request.urlopen(request).read()
    root = lxml.etree.XML(xml_text)
    # TODO http://cangfengzhe.github.io/python/python-lxml.html
    # TODO http://python3-cookbook.readthedocs.io/zh_CN/latest/c06/p03_parse_simple_xml_data.html
    # TODO https://docs.python.org/2/library/xml.etree.elementtree.html
    return PanoramaMetadata_python3(root)


# h_base, w_base should be renamed...
def get_panorama(panoid, zoom):
    h = pow(2, zoom - 1)
    w = pow(2, zoom)
    h_base = 416
    w_base = 512
    panorama = np.zeros((h * w_base, w * w_base, 3), dtype="uint8")
    for y in range(0, h):
        for x in range(0, w):
            img = get_panorama_tile(panoid, zoom, x, y)
            panorama[y * w_base:y * w_base + w_base, x * w_base:x * w_base + w_base, :] = img[0:w_base, 0:w_base, :]
    return panorama[0:h * h_base, 0:w * h_base, :]


# panoid is the value from the panorama metadata
# zoom range is 0->NumZoomLevels inclusively
# x/y range is 0->?
def get_panorama_tile(panoid, zoom, x, y):
    base_uri = 'http://maps.google.com/cbk'
    url = '%s?'
    url += 'output=tile'  # tile output
    url += '&panoid=%s'  # panoid to retrieve
    url += '&zoom=%s'  # zoom level of tile
    url += '&x=%i'  # x position of tile
    url += '&y=%i'  # y position of tile
    url += '&fover=2'  # ???
    url += '&onerr=3'  # ???
    url += '&renderer=spherical'  # standard speherical projection
    url += '&v=4'  # version
    url = url % (base_uri, panoid, zoom, x, y)
    return url_to_image(url)


# Method #1: OpenCV, NumPy, and urllib
# FROM http://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.Request(url)
    xml_text = urllib.request.urlopen(resp).read()
    image = np.asarray(bytearray(xml_text), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def url_to_image_python3(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # return the image
    return img