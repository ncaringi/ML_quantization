import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json

# we use this file to evaluate our results 

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

# Initialize COCO ground truth API
annFile = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\annotations_trainval2017\annotations\instances_val2017.json'
cocoGt = COCO(annFile)

# Initialize COCO detections API
resFile = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_tested\float16\coco_results.json'

# Load the results
with open(resFile, 'r') as f:
    results = json.load(f)

# Load the original size of the images
image_info = {img['id']: (img['width'], img['height']) for img in cocoGt.loadImgs(cocoGt.getImgIds())}

# Rescale Bbox
for result in results:
    image_id = result['image_id']
    orig_width, orig_height = image_info[image_id]
    result['bbox'][0] *= orig_width
    result['bbox'][1] *= orig_height
    result['bbox'][2] *= orig_width
    result['bbox'][3] *= orig_height

# Save the results
recalibrated_resFile = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_tested\float16\recalibrated_coco_results.json'
with open(recalibrated_resFile, 'w') as f:
    json.dump(results, f)

# Load the new results
cocoDt = cocoGt.loadRes(recalibrated_resFile)

# Load the list of image's ID in a json file
image_ids_file = r'C:\Users\Admin\Documents\TU Wien\projets\centernet\val2017_tested\image_ids.json'
with open(image_ids_file, 'r') as f:
    imgIds = json.load(f)

# Running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds  # Evaluate only the specified images
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()