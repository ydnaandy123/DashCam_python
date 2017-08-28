import json

fileDir = "image_info_test-dev2015.json"
with open(fileDir) as meta_file:
    annotations = json.load(meta_file)
    meta_file.close()

result = []
images = annotations['images']
for image in images:
    keypoint = {"image_id": image['id'],
                "category_id": 1,
                "keypoints": [36,181,2,20.25,191,0,35,166,2,20.25,191,0,8,171,2,20.25,191,0,2,246,2,20.25,191,0,20.25,
                              191,0,20.25,191,0,20.25,191,0,20.25,191,0,20.25,191,0,20.25,191,
                              0,20.25,191,0,20.25,191,0,20.25,191,0],
                "score": 0.897}
    result.append(keypoint)
    #break

with open('result.json', 'w') as outfile:
    json.dump(result, outfile)
    outfile.close()
