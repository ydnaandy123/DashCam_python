# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from glumpy import app
from glumpy.graphics.collections import PointCollection
import json

ID = '000034';
with open('src/json/dashcam/deep_match/' + ID + '/pointCloud.json') as data_file:    
    pointCloud = json.load(data_file)

def parsePointClout_SFM(points):
    points_SFM = pointCloud['points']
    print len(points_SFM.keys())
    for key, value in points_SFM.iteritems():
        c = value['color'];
        c.append(255.0);
        points.append(np.array(value['coordinates'])/300.0,
                        color = np.array(c)/255.0,
                        size  = 0.5)   

def parseTrajectory(points):
    trajectory = pointCloud['trajectory']
    isFirst = True
    for key, value in trajectory.iteritems():
        points.append(np.array(value)/300.0,
                        color = np.array([1, 0, 0, 1]),
                        size  = 5)    
    return points

def widowSetting():
    window = app.Window(1024,1024, color=(1,1,1,1))
    @window.event
    def on_draw(dt):
        window.clear()
        points.draw()


    window.attach(points["transform"])
    window.attach(points["viewport"])
    app.run()    

points = PointCollection("agg", color="local", size="local")
parsePointClout_SFM(points);
parseTrajectory(points)
widowSetting();



