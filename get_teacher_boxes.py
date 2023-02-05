import json
from mae.main_linprobe import mae_inference
import mae.models_vit as models_vit
import argparse
import torch
import yaml
import json
from detectron2.config import CfgNode

from ubteacher.modeling import *
from ubteacher.engine import *
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from ubteacher import add_ubteacher_config

from PIL import Image
import numpy as np
import os
from tqdm import tqdm

import torchvision.transforms as T
transform = T.Compose([T.ToTensor()])

def load_mae_model(checkpoint_path):
    mae_model = models_vit.__dict__['vit_base_patch16'](num_classes=2,global_pool=False,)
    checkpoint = torch.load(checkpoint_path)
    checkpoint['model']['head.weight'] = checkpoint['model'].pop('head.1.weight')
    checkpoint['model']['head.bias'] = checkpoint['model'].pop('head.1.bias')
    checkpoint['model'].pop('head.0.running_mean')
    checkpoint['model'].pop('head.0.running_var')
    checkpoint['model'].pop('head.0.num_batches_tracked')
    mae_model.load_state_dict(checkpoint['model'])
    mae_model.to('cuda')

    return mae_model

def load_teacher_model(teacher_checkpoint, config_path):
    config = get_cfg()
    add_ubteacher_config(config)
    config.merge_from_file(config_path)
    config_list = []
    config_list.append('MODEL.WEIGHTS')
    config_list.append(teacher_checkpoint)
    config.merge_from_list(config_list)

    Trainer = UBRCNNTeacherTrainer

    model = Trainer.build_model(config)
    model_teacher = Trainer.build_model(config)
    ensem_ts_model= EnsembleTSModel(model_teacher, model)
    DetectionCheckpointer(ensem_ts_model, save_dir='/output/debug').resume_or_load(config.MODEL.WEIGHTS, resume=False)

    return ensem_ts_model

def inference_models(teacher_checkpoint, config_path, mae_checkpoint, labeled_data_path, instances_path, data_type, save_file_name):
    path_root, labeled, unlabeled = split_labeled_unlabeled(instances_path, labeled_data_path)
    if mae_checkpoint:
        mae_model = load_mae_model(mae_checkpoint)
    teacher_model = load_teacher_model(teacher_checkpoint, config_path)
    teacher_model.eval()
    labeled_save_filename = save_file_name
    unlabeled_save_filename = save_file_name.split('labeled')[0] + 'unlabeled.json'
    if data_type == 'both' or data_type == 'labeled':
        all_outputs_on_labeled = {'images': [], 'bboxes': [], 'scores': [], 'labels': [], 'mae_scores': []}
        for image_path in labeled:
            model_ckpt = None
            all_outputs_on_labeled['images'].append(image_path)
            bboxes, scores, labels = get_teacher_predictions_for_image(teacher_model, image_path)
            if len(bboxes):
                all_outputs_on_labeled['bboxes'].append(bboxes)
                all_outputs_on_labeled['scores'].append(scores)
                all_outputs_on_labeled['labels'].append(labels)

                bboxes = torch.from_numpy(np.array(bboxes))
                scores = torch.from_numpy(np.array(scores))
                labels = torch.from_numpy(np.array(labels))
                per_image_prediction = (0, torch.unsqueeze(bboxes,0), torch.unsqueeze(scores, 0), torch.unsqueeze(labels, 0))
                if mae_checkpoint:
                    model_ckpt = mae_model
                    mae_scores = get_mae_model_scores(per_image_prediction, model_ckpt, image_path)
                    mae_scores = mae_scores[0].detach().cpu().numpy()
                    all_outputs_on_labeled['mae_scores'].append(mae_scores.tolist())
               
            else:
                all_outputs_on_labeled['bboxes'].append([])
                all_outputs_on_labeled['scores'].append([])
                all_outputs_on_labeled['labels'].append([])
                all_outputs_on_labeled['mae_scores'].append([])

        with open(labeled_save_filename, 'w') as jsonf:
            json.dump(all_outputs_on_labeled, jsonf)

    if data_type == 'both' or data_type == 'unlabeled':
        all_outputs_on_unlabeled = {'images': [], 'bboxes': [], 'scores': [], 'labels': [], 'mae_scores': []}
        for unlab_image_path in tqdm(unlabeled):
            all_outputs_on_unlabeled['images'].append(unlab_image_path)
            bboxes, scores, labels = get_teacher_predictions_for_image(teacher_model, unlab_image_path)
            if len(bboxes):
                all_outputs_on_unlabeled['bboxes'].append(bboxes)
                all_outputs_on_unlabeled['scores'].append(scores)
                all_outputs_on_unlabeled['labels'].append(labels)

                bboxes = torch.from_numpy(np.array(bboxes))
                scores = torch.from_numpy(np.array(scores))
                labels = torch.from_numpy(np.array(labels))
                per_image_prediction = (0, torch.unsqueeze(bboxes,0), torch.unsqueeze(scores, 0), torch.unsqueeze(labels, 0))
                if mae_checkpoint:
                    mae_scores = get_mae_model_scores(per_image_prediction, mae_model, unlab_image_path)
                    mae_scores = mae_scores[0].detach().cpu().numpy()
                    all_outputs_on_unlabeled['mae_scores'].append(mae_scores.tolist())
                
            else:
                all_outputs_on_unlabeled['bboxes'].append([])
                all_outputs_on_unlabeled['scores'].append([])
                all_outputs_on_unlabeled['labels'].append([])
                all_outputs_on_unlabeled['mae_scores'].append([])

        with open(unlabeled_save_filename, 'w') as jsonf:
            json.dump(all_outputs_on_unlabeled, jsonf)
        

def get_teacher_predictions_for_image(teacher_model, image_path):
    with open(image_path, 'rb') as image:
        im = Image.open(image)
        im = im.convert("RGB")
        tensor_image = np.array(im)
        if len(tensor_image.shape) == 2:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        tensor_image = torch.Tensor(tensor_image.transpose((2,0,1)))
        input_image = {'image': tensor_image}
        if 'coco' in image_path:
            with torch.no_grad():
                teacher_proposals_roih_unsup_k = teacher_model.modelTeacher([input_image])
        else:
            with torch.no_grad():
                teacher_proposals_roih_unsup_k = teacher_model.modelStudent([input_image])

        if len(teacher_proposals_roih_unsup_k):
            fields = teacher_proposals_roih_unsup_k[0]['instances'].get_fields()
            pred_boxes = fields['pred_boxes'].tensor.data.cpu().detach().numpy()
            pred_classes = fields['pred_classes'].data.cpu().detach().numpy()
            pred_scores = fields['scores'].data.cpu().detach().numpy()
            bboxes = []
            scores = []
            classes = []
            for k in range(len(pred_boxes)):
                box = [pred_boxes[k][0].item(), pred_boxes[k][1].item(), pred_boxes[k][2].item(), pred_boxes[k][3].item()]
                bboxes.append(box)
                classes.append(pred_classes[k].item())
                scores.append(pred_scores[k].item())

            return bboxes, scores, classes

        return [], [], []

def get_mae_model_scores(per_image_prediction, mae_model, image_path):
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert("RGB")
        tensor_image = np.array(image)  # otherwise nothing will work! but should we transpose this?
    if len(tensor_image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    tensor_image = transform(image)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    tensor_image = tensor_image.cuda()

    _, _, _, _, output = mae_inference(per_image_prediction, mae_model, tensor_image)

    return output

def split_labeled_unlabeled(dataset, labeled_data_path):
    labeled_filenames = []
    unlabeled_filenames = []
    with open(dataset) as data_path:
        train_data = json.load(data_path)
    images = train_data['images']
    
    with open(labeled_data_path) as labeled_path:
        labeled_path = list(labeled_path)
    
    path_root = ''
    for i in labeled_path[0].split('/')[:-1]:
        path_root += i +'/'

    for image in images:
        if 'coco' in path_root:
            filename = path_root + image['file_name']
        filename = image['file_name']
        if os.path.exists(filename):
            filename += '\n'
            if filename in labeled_path:
                labeled_filenames.append(filename[:-1])
            else:
                unlabeled_filenames.append(filename[:-1])

    return path_root, labeled_filenames, unlabeled_filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', default='')
    parser.add_argument('--config_path', default='')
    parser.add_argument('--mae_path', default=None)
    parser.add_argument('--labeled_data_path', default='')
    parser.add_argument('--instances_path', default='')
    parser.add_argument('--data_type', default='labeled')
    parser.add_argument('--save_file_name_for_labeled', default='')
    args = parser.parse_args()

    inference_models(args.teacher_path, args.config_path, args.mae_path, args.labeled_data_path, args.instances_path, args.data_type, args.save_file_name_for_labeled)