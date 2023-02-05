import os
from tqdm import tqdm
import json
from PIL import Image


def change_to_coco_format(phase_folder, dataset_name, stage, testing_file_path):
    dataset_label_root = '{}/labels/'.format(phase_folder)

    class_names_filepath = '{}/stage0.json'.format(phase_folder)
    with open(class_names_filepath, "r") as class_names:
        classes = json.load(class_names)

    with open('{}/train_0.txt'.format(phase_folder)) as f:
        line = f.readline()
    image_folder = line.replace(line.split('/')[-1], '')
    train_images = os.listdir(image_folder)

    with open(testing_file_path) as t:
        test_image_names = t.readlines()

    data_folder = image_folder.replace(image_folder[:-1].split('/')[-1], '')[:-1]
    dataset_folder = 'datasets/{}_{}'.format(dataset_name, stage)
    os.makedirs(dataset_folder)

    dataset_train_images_folder = '{}/train'.format(dataset_folder)
    dataset_test_images_folder = '{}/test'.format(dataset_folder)

    test_folder = '{}test'.format(data_folder)

    if not os.path.exists(dataset_train_images_folder):
        os.symlink(image_folder[:-1], dataset_train_images_folder)

    if not os.path.exists(dataset_test_images_folder):
        os.symlink(test_folder, dataset_test_images_folder)

    annotations_folder = '{}/annotations'.format(dataset_folder)
    if not os.path.exists(annotations_folder):
        print('___creating annotations folder___')
        os.makedirs(annotations_folder, mode=0o777)

    test_images = {'images': [], 'categories' : [{'supercategory': classes[k], 'id': int(k), 'name': classes[k]} for k in classes.keys()]}
    dataset_annotations = {'images': [], 'annotations': [], 'categories' : [{'supercategory': classes[k], 'id': int(k), 'name': classes[k]} for k in classes.keys()]}
    bbox_id = 0
    image_id = 0
    
    print("________Formating dataset to COCO format________")
    for train_image in tqdm(train_images):
        filename = image_folder + train_image
        im = Image.open(filename)
        width, height = im.size
        dataset_annotations['images'].append({'file_name': filename, 'height': height,'width': width,'id': image_id})
        label_path = dataset_label_root + train_image.split('.')[0] + '.txt'
        if os.path.exists(label_path):
            with open(label_path) as label:
                for line in label:
                    ann = line.split(' ')
                    width = int(ann[3]) - int(ann[1])
                    height = int(ann[4]) - int(ann[2])
                    area = width * height
                    dataset_annotations['annotations'].append({'image_id': image_id, 'category_id': int(ann[0]), 'bbox': [int(ann[1]), int(ann[2]), width, height], 'id': bbox_id, 'iscrowd': 0, 'area': area})
                    bbox_id += 1

        image_id += 1

    for test_image in tqdm(test_image_names):
        im = Image.open(test_image[:-1])
        width, height = im.size
        test_images['images'].append({'file_name': test_image[:-1], 'height': height,'width': width,'id': image_id})
        image_id += 1

    train_instances_json = 'datasets/{}_{}/annotations/instances_train.json'.format(dataset_name, stage)
    if os.path.exists(train_instances_json):
        os.remove(train_instances_json)
    with open(train_instances_json, 'w') as train_json:
        json.dump(dataset_annotations, train_json)

    test_instances_json = 'datasets/{}_{}/annotations/instances_test.json'.format(dataset_name, stage)
    if os.path.exists(test_instances_json):
        os.remove(test_instances_json)
    with open(test_instances_json, 'w') as test_json:
        json.dump(test_images, test_json)
    
