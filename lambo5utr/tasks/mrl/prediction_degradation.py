import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def predict_in_vitro_degradation(
    str_array,
    max_seq_len=50,
    upstream_seq="",
    downstream_seq="",
    project_root="",
):

    tmp_path = os.path.join(project_root, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    input_file = os.path.join(tmp_path, "degradation_input.txt")
    output_file = os.path.join(tmp_path, "degradation_output.txt")
    exp_condition = "deg_Mg_pH10"
    nullrecurrent_path = os.path.join(project_root, "misc/run_nullrecurrent.py")

    # record sequences into .txt
    with open(input_file, "w") as f:
        for seq in str_array:
            seq = _format_sequence(seq, max_seq_len, upstream_seq, downstream_seq)
            f.write(seq + "\n")
    
    # clear output file to make sure the prediction is based on the input
    open(output_file, "w").close()

    # python scripts/nullrecurrent_inference.py -d deg -i example.txt -o predict.txt
    # run Nullrecurrent
    option_args = {
        "d":exp_condition,   # default: deg_Mg_pH10 choice from "reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"
        "i":input_file,         # input file
        "o":output_file,        # output file
    }

    cmd = "python3 {}".format(nullrecurrent_path)
    for k, v in option_args.items():
        cmd += " -{} {}".format(k, v)
    logger.info("executing Nullrecurrent: {}".format(cmd))
    os.system(cmd)

    degradation_scores = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line)>0:
                line = line.split(", ")
                score = np.mean(np.array(line).astype(float))
                degradation_scores.append(score)

    return np.array(degradation_scores).reshape(-1,1)

# prediction function for multi-processing
def predict_in_vitro_degradation_multi(
    str_array,
    max_seq_len=50,
    upstream_seq="",
    downstream_seq="",
    exp_name=None,
    project_root="",
):

    tmp_path = os.path.join(project_root, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    input_file = os.path.join(tmp_path, f"degradation_input_{exp_name}.txt")
    output_file = os.path.join(tmp_path, f"degradation_output_{exp_name}.txt")
    exp_condition = "deg_Mg_pH10"
    nullrecurrent_path = os.path.join(project_root, "misc/run_nullrecurrent.py")

    # record sequences into .txt
    with open(input_file, "w") as f:
        for seq in str_array:
            seq = _format_sequence(seq, max_seq_len, upstream_seq, downstream_seq)
            f.write(seq + "\n")

    # clear output file to make sure the prediction is based on the input
    open(output_file, "w").close()

    # python scripts/nullrecurrent_inference.py -d deg -i example.txt -o predict.txt
    # run Nullrecurrent
    option_args = {
        "d": exp_condition,  # default: deg_Mg_pH10 choice from 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C'
        "i": input_file,  # input file
        "o": output_file,  # output file
    }

    cmd = "python3 {}".format(nullrecurrent_path)
    for k, v in option_args.items():
        cmd += " -{} {}".format(k, v)
    logger.info("executing Nullrecurrent: {}".format(cmd))
    os.system(cmd)

    degradation_scores = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.split(", ")
                score = np.mean(np.array(line).astype(float))
                degradation_scores.append(score)

    return np.array(degradation_scores).reshape(-1, 1)

def _format_sequence(seq, max_seq_len, upstream_seq, downstream_seq):
    assert len(seq) <= max_seq_len
    seq = upstream_seq + seq + downstream_seq
    seq = seq.upper()
    seq = seq.replace("T", "U")
    return seq