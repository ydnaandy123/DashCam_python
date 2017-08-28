import json

fileDir = "person_keypoints_test-dev2015_hourglass_results.json"
with open(fileDir) as meta_file:
    annotations = json.load(meta_file)
    meta_file.close()

