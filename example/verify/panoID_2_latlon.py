# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------

import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import google_store


id = 'z0KlatCyk1LcjQHQ1KzGZg'
pano = google_store.get_panorama_metadata(panoid=id)
print(pano.PanoId, pano.Lat, pano.Lon)

