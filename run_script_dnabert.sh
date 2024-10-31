#!/bin/sh
TRIAL=0
DEVICE_ID=0
INIT_ID=3
SEED=1
export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export TF_FORCE_GPU_ALLOW_GROWTH=false
module load R

python3 scripts/black_box_opt.py seed=$SEED trial_id=$TRIAL device_id=$DEVICE_ID exp_name=exp_mugd_bert_with_tunedhp_unmod1 task=task_mugd_cnn task.init_randomize=False task.given_filename=Sample2019_unmod1_initset${INIT_ID} task.num_start_examples=512 optimizer=lambo tokenizer=dnapretrained encoder=dnabertonly encoder.mask_ratio=0.2 optimizer.lr=0.3 optimizer.entropy_penalty=0.3 optimizer.num_rounds=64 optimizer.num_gens=16 optimizer.num_opt_steps=32 optimizer.mutation_method=mask_annealing optimizer.annealing_epochs=16 optimizer.replacement_method=mask_dominant
