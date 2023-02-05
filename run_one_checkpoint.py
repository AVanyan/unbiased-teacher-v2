from pipeline_utils import *
import os
import shutil

from dataset_to_coco_format import change_to_coco_format
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

# hacky way to register
from ubteacher.modeling import *
from ubteacher.engine import *
from ubteacher import add_ubteacher_config
import wandb

from detectron2.data.datasets import register_coco_instances

import argparse

def main(args):

    labeled_file_path = './session_data/{}/{}/train_{}.txt'.format(args.session_id, args.phase, args.stage)
    unlabeled_file_path = './session_data/{}/{}/train_unlabeled_{}.txt'.format(args.session_id, args.phase, args.stage)
    print(labeled_file_path)

    with open(labeled_file_path) as fp:
        len_of_labeled = len(fp.readlines())

    with open(unlabeled_file_path) as ufp:
        len_of_unlabeled = len(ufp.readlines())

    all_data_len = len_of_labeled + len_of_unlabeled

    dataset_folder = './datasets/{}_{}'.format(args.dataset_name, args.stage)

    train_json = '{}/annotations/instances_train.json'.format(dataset_folder)
    test_json = '{}/annotations/instances_test.json'.format(dataset_folder)
    print('___registering_dataset___')
    register_coco_instances("{}_train".format(args.dataset_name), {}, train_json, '{}/train'.format(dataset_folder))
    register_coco_instances("{}_test".format(args.dataset_name), {}, test_json, '{}/test'.format(dataset_folder))

    steps_limit = 50000
    if len_of_labeled/all_data_len > 0.7:
        burn_up_steps = 10000
        steps = 5000
    else:
        burn_up_steps = max(min(int(12.5*len_of_labeled), 2000), 250)
        steps = max(min(500*len_of_labeled,  steps_limit - burn_up_steps), 8000)

    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file('./configs/Faster-RCNN/evaluation/base_config.yaml')
    config_list = []
    config_list.append('LABELED_FILE_PATH')
    config_list.append(labeled_file_path)
    config_list.append('MODEL.ROI_HEADS.NUM_CLASSES')
    config_list.append(args.class_num)
    config_list.append('DATASETS.TRAIN')
    config_list.append(('{}_train'.format(args.dataset_name),))
    config_list.append('DATASETS.TEST')
    config_list.append(('{}_test'.format(args.dataset_name),))
    config_list.append('SEMISUPNET.BURN_UP_STEP')
    config_list.append(burn_up_steps)
    config_list.append('SOLVER.STEPS')
    config_list.append((steps,))
    config_list.append('SOLVER.MAX_ITER')
    config_list.append(steps)

    argsdict = vars(args)

    if 'experiment_name' in argsdict and argsdict['experiment_name'] is not None:
        config_list.append('EXPERIMENT_NAME')
        config_list.append(argsdict['experiment_name'])
        config_list.append('OUTPUT_DIR')
        config_list.append('./output/{}_{}'.format(argsdict['experiment_name'], args.session_id))
    
    if 'batch_size' in argsdict and argsdict['batch_size'] is not None:
        config_list.append('SOLVER.IMG_PER_BATCH_LABEL')
        config_list.append(argsdict['batch_size'])
        config_list.append('IMG_PER_BATCH_UNLABEL')
        config_list.append(argsdict['batch_size'])

    cfg.merge_from_list(config_list)
    cfg.freeze()
    default_setup(cfg, args)
    wandb.init(project='{}_{}_{}_{}'.format(args.dataset_name, args.phase, args.stage, cfg.EXPERIMENT_NAME), sync_tensorboard=True,
           settings=wandb.Settings(start_method="thread", console="off"), config=cfg, )
    wandb.run.name = cfg.EXPERIMENT_NAME

    Trainer = UBRCNNTeacherTrainer

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        Trainer.test(cfg, model)

    else:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)

        trainer.train()
    
    results_json_path = '{}/inference/coco_instances_results.json'.format(cfg.OUTPUT_DIR)
    test_instances = './datasets/{}_{}/annotations/instances_test.json'.format(args.dataset_name, args.stage)
    Trainer.json_to_csv(cfg, results_json_path, test_instances)

if __name__ == "__main__":
    def_args = default_argument_parser()
    
    parser = argparse.ArgumentParser(parents=[def_args], description="UBTeacher training", add_help=False)

    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--session_id', type=str, default=None)
    parser.add_argument('--phase', type=str, default=None)
    parser.add_argument('--stage', type=int, default=None)
    parser.add_argument('--class_num', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--teacher_init_path', type=str, default=None)
    parser.add_argument('--skip_data_path', type=str, default=None)
    parser.add_argument('--teacher_init_skip_last_layer', action='store_true')
    parser.add_argument('--oracle_train_on_labeled', action='store_true')
    parser.add_argument('--skip_burn_in', action='store_true')  # if true, student will start immediately
    parser.add_argument('--gradac_batches', type=int, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--consistency_lambda', type=float, default=None)
    parser.add_argument('--oracle_iou_threshold', type=float, default=None)
    parser.add_argument('--lr_schedule', type=str, default=None)
    parser.add_argument('--oracle_model_path', type=str, default=None)
    parser.add_argument('--oracle_model_path_init', type=str, default=None)
    parser.add_argument('--oracle_pretrained', type=bool, default=None)
    parser.add_argument('--student_learning_rate', type=float, default=None)
    parser.add_argument('--student_lr_schedule', type=str, default=None)
    parser.add_argument('--oracle_feature_data_path', type=str, default=None)
    parser.add_argument('--student_warmup_steps', type=int, default=None)
    parser.add_argument('--gradient_clip_threshold', type=float, default=None)
    parser.add_argument('--confidence_threshold', type=float, default=None)
    parser.add_argument('--box_score_thresh', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--EMA_keep_rate', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--augmentation', type=int, default=None)
    parser.add_argument('--initialization', type=str, default=None)
    parser.add_argument('--reuse_classifier', type=str, default=None)
    parser.add_argument('--check_val_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--thresholding_method', type=str, default=None)
    parser.add_argument('--dt_gamma', type=float, default=None)
    parser.add_argument('--total_steps_teacher_initial', type=int, default=None)
    parser.add_argument('--total_steps_student_initial', type=int, default=None)
    parser.add_argument('--inference_only', action='store_true')  # do not train. attempt to test with the checkpoint
    parser.add_argument('--test_with_student', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None)

    args = parser.parse_args()
    print("Command Line Args:", args)

    training_dataset_name = "api_dataset_train"
    phase_folder = './session_data/{}/{}'.format(args.session_id, args.phase)
    label_root = '{}/labels'.format(phase_folder)

    class_names = [i for i in range(args.class_num)]
    class_names_str = []
    for class_name in class_names:
        if type(class_name) == int:
            class_name = str(class_name)
        class_names_str.append(class_name)

    test_dataset_name = "api_dataset_test"
    testing_file_path = '{}/test.txt'.format(phase_folder)

    dataset_folder = './datasets/{}_{}'.format(args.dataset_name, args.stage)
    if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
        print("deleting dataset folder")
        shutil.rmtree(dataset_folder)
    change_to_coco_format(phase_folder, args.dataset_name, args.stage, testing_file_path)        

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    model_output_csv = os.path.join('./output/{}_{}'.format(args.experiment_name, args.session_id), 'output.csv')
    if os.path.exists(args.output_csv):
        new_name = args.output_csv + '.tmp'
        print("{} already exists. Moving it to {}".format(args.output_csv, new_name))
        if os.path.exists(new_name):
            print("Removing old {}".format(new_name))
            os.remove(new_name)
        shutil.move(args.output_csv, new_name)
    print("Copying {} to {}".format(model_output_csv, args.output_csv))
    shutil.copy(model_output_csv, args.output_csv)