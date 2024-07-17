import os
from glob import glob
import shutil
from tqdm import tqdm
import cv2
import json
import random
random.seed(0)
# 0 
with open('/home/nute11a/workspace/SAMUS/configs/bbox-BreastCancer.json', 'r') as f:
    bboxdata = json.load(f)


saveRoot = '/home/nute11a/workspace/SAMUS/dataset/BreastCancer_US_COCO'
os.makedirs(f'{saveRoot}/train2023', exist_ok=True)
os.makedirs(f'{saveRoot}/val2023', exist_ok=True)
os.makedirs(f'{saveRoot}/test2023', exist_ok=True)

for mode in ['train2023', 'val2023', 'test2023']:
    #! 1) Images file 나누기
    ids = [id_.strip() for id_ in open(f"/home/nute11a/workspace/SAMUS/dataset/BreastCancer_US/{mode.replace('2023', '')}-BreastCancer_US.txt")]
    for id in tqdm(ids):
        class_id0, sub_path, filename = id.split('/')
        shutil.copy(f'/home/nute11a/workspace/SAMUS/dataset/BreastCancer_US/img/{filename}', f'{saveRoot}/{mode}/{filename}')

    #! 2) Json Annotation file
    json_dict = {}
    ### info
    json_dict['info'] = {
        'description' : 'Ultrasound Detection Dataset'
    }
    ### licenses
    json_dict['licenses'] = []
    ### images & annotations
    json_dict['images'] = []
    json_dict['annotations'] = []
    seg_idx = random.randint(2000, 3000)
    for img_idx, id in enumerate(ids):
        class_id0, sub_path, filename = id.split('/')
        img = cv2.imread(f'{saveRoot}/{mode}/{filename}')
        h, w, _ = img.shape
        json_dict['images'].append({
            'file_name' : filename,
            "height"    : h,
            "width"     : w,
            "id"        : img_idx
        })
        # json
        bboxes = bboxdata[filename.replace('.png', '')]
        for bbox in bboxes:
            json_dict['annotations'].append({
                'id'            : seg_idx,      # img_idx
                'category_id'   : id,
                'iscrowd'       : 0,
                'image_id'      : img_idx,
                'bbox'          : [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]      # minx,miny,maxx,maxy -> minx,miny,w,h
            })
            seg_idx += 1
            
    ### categories
    json_dict['categories'] = [
        {
            "supercategory" : "tumor",
            "id"            : 1,
            "name"          : ""
        }
    ]
    print(' =========== end =========== ')
    with open(f'/home/nute11a/workspace/SAMUS/dataset/BreastCancer_US_COCO/annotations/US_Bbox_{mode}.json', 'w') as f:
        json.dump(json_dict, f, indent=3)