import json
import scipy.io as sio
import sys
sys.path.append('../Refference/')
import streetview_my

from pprint import pprint

with open('json/metaOnly.json') as data_file:    
    data = json.load(data_file)


print data[str(0)]['panoId']

panoDict = {}

cur = 0
pano = streetview_my.GetPanoramaMetadata(data[str(0)]['panoId'])
pano_for_mat = {'Lat':pano.Lat, 'Lon':pano.Lon, 'panoId':pano.PanoId, 'AnnotationLinks':pano.AnnotationLinks}
panoDict['street' + str(cur)] = pano_for_mat
cur += 1

panoMeta = {}
panoMeta['len'] = len(panoDict.keys())	
panoMeta['data'] = panoDict		

sio.savemat('streetview_set_onlyMeta_G1.mat', panoMeta)