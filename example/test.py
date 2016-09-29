# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
import sys

sys.path.append('/home/andy/Documents/gitHub/DashCam_python/module')  # use the module under 'module'
import file_process
import google_store
id = 'uSjqj9Lt256V8I7RckMykA'
pano = google_store.get_panorama_metadata(panoid=id)
print(pano.id)