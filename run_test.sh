#!/bin/sh
TRIAL=0
DEVICE_ID=0
INIT_ID=3
SEED=1
export CUDA_VISIBLE_DEVICES=$DEVICE_ID
export TF_FORCE_GPU_ALLOW_GROWTH=false
module load R

echo -e "\n=================================================="
echo "=================================================="
echo "LaMBO-DNABERT Test 1/4: Checking predictors"
echo "=================================================="
echo -e "==================================================\n"
python3 ./python_scripts/test_predictors.py

echo -e "\n=================================================="
echo "=================================================="
echo "LaMBO-DNABERT Test 2/4: Checking DNABERT"
echo "=================================================="
echo -e "==================================================\n"

python3 ./python_scripts/test_dnabert.py seed=$SEED trial_id=$TRIAL device_id=$DEVICE_ID exp_name=exp_test task=regex_dna optimizer=lambo tokenizer=dnapretrained encoder=dnabertonly

echo -e "\n=================================================="
echo "=================================================="
echo "LaMBO-DNABERT Test 3/4: Checking LaMBO"
echo "This step would take a few minutes."
echo "=================================================="
echo -e "==================================================\n"

python3 ./python_scripts/test_lambo.py seed=$SEED trial_id=$TRIAL device_id=$DEVICE_ID exp_name=exp_test task=regex_dna optimizer=lambo tokenizer=dna encoder=mlm_cnn surrogate=multi_task_exact_gp_cnn surrogate.num_epochs=8 optimizer.num_rounds=1 optimizer.num_gens=3 optimizer.num_opt_steps=2 wandb_mode=offline

echo -e "\n=================================================="
echo "=================================================="
echo "LaMBO-DNABERT Test 4/4: Checking All"
echo "This step would take 10-20 minutes."
echo "=================================================="
echo -e "==================================================\n"

python3 ./python_scripts/test_lambo.py seed=$SEED trial_id=$TRIAL device_id=$DEVICE_ID exp_name=exp_test task=task_mugd_cnn optimizer=lambo tokenizer=dnapretrained encoder=dnabertonly surrogate=multi_task_exact_gp surrogate.num_epochs=8 optimizer.num_rounds=1 optimizer.num_gens=3 optimizer.num_opt_steps=2 wandb_mode=offline task.given_filename=Sample2019_unmod1_initset${INIT_ID}
