'''
#!/usr/bin/python2
import sys
sys.path.append('module')	# use the module under 'module'
import google_store

# Create PanoFetcher and dashCamFileProcess
zoom, radius = 1, 30
panofetcher = google_store.PanoFetcher(zoom, radius)
# D1: large google date downloading
# no idead what's going on...
panofetcher.BFS_aug('Civic_Boulevard_forDemo', (25.0446129,121.5401195), 10)
# some core data missing (ex:depth)
#panofetcher.BFS_aug('Civic_Boulevard_forDemo', (25.0446129,121.5401195), 10)
'''

#!/usr/bin/python3
import sys
sys.path.append('module')	# use the module under 'module'
import google_parse
import base_process
# D2: The World Geodetic System (WGS) is a standard 
#     or use in cartography, geodesy, and navigation including by GPS. 
#     It comprises a standard coordinate system for the Earth
# base_process
fileID = 'Civic_Boulevard'
streetView3DRegion = google_parse.StreetView3DRegion(fileID)
topology = streetView3DRegion.createTopoloy()
print (topology['a_position'])
# D3: different_axis
# glumpy_5_globalSV3D

# D4: car_box
# glumpy_0_drawPtCloud_sfm















# D3: ptCloud construct time
#glumpy_1_drawPtCloud_streetview

# D4: kd-tree
#

# lack of demo...
#senmantic segment
#generate random viewPoint
#car on the street