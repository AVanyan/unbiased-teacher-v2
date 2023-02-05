import json
from tqdm import tqdm
import random

nightowls_labeled_images_path = '/mnt/lwll/lwll-coral/hrant/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/train_3.txt'
nightowls_labeled_images = []
with open(nightowls_labeled_images_path) as no:
    nightowls_labeled_images = no.readlines()


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def randlevel(a, b, l, side):
    return random.randint(a, b)
    if a > b:
        return b
    if side == 'a':
        x = b
    else:
        x = a
    while l > 0:
        if side == 'a':
            x = random.randint(a, x)
        else:
            x = random.randint(x, b)
        l -= 1
    return x

# f = open('/lwll/external/coco_2017/annotations/instances_train2017.json')
f = open('/lwll/external/nightowls/annotations/instances_train.json')

data = json.load(f)
lines = []

popok = []
hw = {}
imname = {}

for img in data['images']:
    hw[img['id']] = img['height'], img['width']
    imname[img['id']] = img['file_name']

for annotation in tqdm(data['annotations']):
    if imname[annotation['image_id']] + '\n' in nightowls_labeled_images:
        strr = imname[annotation['image_id']] + ' '
        w, h = hw[annotation['image_id']]

        xmin = int(annotation['bbox'][0])
        ymin = int(annotation['bbox'][1])
        xmax = xmin + int(annotation['bbox'][2])
        ymax = ymin + int(annotation['bbox'][3])

        # print(xmin, xmax, ymin, ymax, h, w)
        xmin1 = randlevel(max(0, xmin - 8), min(xmax, xmin + 8), 3, 'b')
        xmax1 = randlevel(max(xmin1, xmax - 8), min(xmax + 8, h), 3, 'a')

        ymin1 = randlevel(max(0, ymin - 8), min(ymax, ymin + 8), 3, 'b')
        ymax1 = randlevel(max(ymin1, ymax - 8), min(ymax + 8, w), 3, 'a')

        conf = bb_intersection_over_union([xmin, ymin, xmax, ymax], [xmin1, ymin1, xmax1, ymax1])
        # print(conf)
        if xmin1 != xmax1 and ymin1 != ymax1: 
            strr = strr + '[' + str(xmin1) + ',' + str(ymin1) + ',' + str(xmax1) + ',' + str(ymax1) + '] ' + str(conf) + '\n'
            lines.append(strr)

f = open("/home/hkhachatrian/unbiased_teacher/unbiased-teacher-v2/objectness_data/nightowls_labeled_debug.txt", "a+")
f.writelines(lines)
f.close()