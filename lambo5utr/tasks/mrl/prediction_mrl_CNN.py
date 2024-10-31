import numpy as np
import gc
from tensorflow import keras
import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)

def predict_mrl_CNN(
    str_array,
    path_to_model,
    max_seq_len=50,
    do_scale=False,
    batch_size=2,
    num_gpu=0,
    device="cuda",
    project_root="",
):
    tmp_path = os.path.join(project_root, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    input_file = os.path.join(tmp_path, "mrl_cnn_input.txt")
    output_file = os.path.join(tmp_path, "mrl_cnn_output.txt")
    mrl_cnn_path = os.path.join(project_root, "lambo5utr/tasks/mrl/Sample2019_CNN.py")
    
    # record sequences into .txt
    with open(input_file, "w") as f:
        for seq in str_array:
            f.write(seq + "\n")
    # clear output file to make sure the prediction is based on the input
    open(output_file, "w").close()

    option_args = {
        "input": input_file,  # input file
        "output": output_file,  # output file
        "path_model": os.path.join(project_root, path_to_model),
    }
    cmd = "python3 {}".format(mrl_cnn_path)
    for k, v in option_args.items():
        cmd += " --{} {}".format(k, v)
    logger.info("executing MRL prediction: {}".format(cmd))
    os.system(cmd)

    preds = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            preds.append(float(line))

    return np.array(preds)

# prediction function for multi-processing
def predict_mrl_CNN_multi(
    str_array,
    path_to_model,
    max_seq_len=50,
    do_scale=False,
    batch_size=2,
    num_gpu=0,
    device="cuda",
    exp_name=None,
    project_root="",
):
    tmp_path = os.path.join(project_root, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    input_file = os.path.join(tmp_path, f"mrl_cnn_input_{exp_name}.txt")
    output_file = os.path.join(tmp_path, f"mrl_cnn_output_{exp_name}.txt")
    mrl_cnn_path = os.path.join(project_root, "lambo5utr/tasks/mrl/Sample2019_CNN.py")
    # record sequences into .txt
    with open(input_file, "w") as f:
        for seq in str_array:
            f.write(seq + "\n")
    # clear output file to make sure the prediction is based on the input
    open(output_file, "w").close()

    option_args = {
        "input": input_file,  # input file
        "output": output_file,  # output file
        "path_model": os.path.join(project_root, path_to_model),
    }
    cmd = "python3 {}".format(mrl_cnn_path)
    for k, v in option_args.items():
        cmd += " --{} {}".format(k, v)
    logger.info("executing MRL prediction: {}".format(cmd))
    os.system(cmd)

    preds = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            preds.append(float(line))

    return np.array(preds)