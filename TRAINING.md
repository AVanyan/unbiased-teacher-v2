We generate training data for the objectness network using the following command for COCO:
python /home/hkhachatrian/unbiased_teacher/unbiased-teacher-v2/make_feature_files.py
This stores the data (COCO objectness data) at:
/mnt/lwll/lwll-coral/hrant/objectness_data/coco_synthetic_for_mae.txt

For Nightowls:
python /home/hkhachatrian/unbiased_teacher/unbiased-teacher-v2/make_feature_files.py #TODO change command
This stored the data (nightowls stage 3 objectness data) at:
/mnt/lwll/lwll-coral/hrant/objectness_data/nightowls_labeled_synthetic.txt


We finetune pretrained MAE on COCO objectness data using the following command:
—-------
This stores the model (COCO Objectness Model) at PATH

We finetune COCO Objectness Model further on Nightowls stage 3 objectness data using the command:
—----------
This stores the model (Nightowls Objectness Model) at PATH 
