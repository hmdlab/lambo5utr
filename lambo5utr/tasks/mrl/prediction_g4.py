import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def predict_g4(
    str_array,
    max_seq_len=50,
    upstream_seq="",
    downstream_seq="",
    step_size=20,
    threshold=0.0,
    project_root="",
):
    # expected sequence length by DeepG4
    LEN_DEEPG4 = 201
    deepg4_path = os.path.join(project_root, "misc/run_deepg4.R")
    tmp_path = os.path.join(project_root, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    input_file = os.path.join(tmp_path, "g4_input.txt")
    output_file = os.path.join(tmp_path, "g4_output.txt")
    # clear output file to make sure the prediction is based on the input
    open(output_file, "w").close()

    # record sequences into .txt
    with open(input_file, "w") as f:
        for seq in str_array:
            seq = _format_sequence(
                seq, max_seq_len, upstream_seq, downstream_seq, LEN_DEEPG4
            )
            f.write(seq + "\n")

    # run DeepG4
    option_args = {
        "i": input_file,  # input file
        "o": output_file,  # output file
        "k": step_size,  # step size
        "t": threshold,  # threshold G4 score. sequences with >threshold will be kept in the output.
    }
    cmd = "Rscript {}".format(deepg4_path)
    for k, v in option_args.items():
        cmd += " -{} {}".format(k, v)
    logger.info("executing DeepG4: {}".format(cmd))
    os.system(cmd)

    g4_scores = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            g4_scores.append(float(line))

    return np.array(g4_scores).reshape(-1, 1)

# prediction function for multi-processing
def predict_g4_multi(
    str_array,
    max_seq_len=50,
    upstream_seq="",
    downstream_seq="",
    step_size=20,
    threshold=0.0,
    exp_name=None,
    project_root="",
):
    # expected sequence length by DeepG4
    LEN_DEEPG4 = 201
    deepg4_path = os.path.join(project_root, "misc/run_deepg4.R")
    tmp_path = os.path.join(project_root, "tmp")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    input_file = os.path.join(tmp_path, f"g4_input_{exp_name}.txt")
    output_file = os.path.join(tmp_path, f"g4_output_{exp_name}.txt")
    # clear output file to make sure the prediction is based on the input
    open(output_file, "w").close()

    # record sequences into .txt
    with open(input_file, "w") as f:
        for seq in str_array:
            seq = _format_sequence(
                seq, max_seq_len, upstream_seq, downstream_seq, LEN_DEEPG4
            )
            f.write(seq + "\n")

    # run DeepG4
    option_args = {
        "i": input_file,  # input file
        "o": output_file,  # output file
        "k": step_size,  # step size
        "t": threshold,  # threshold G4 score. sequences with >threshold will be kept in the output.
    }
    cmd = "Rscript {}".format(deepg4_path)
    for k, v in option_args.items():
        cmd += " -{} {}".format(k, v)
    logger.info("executing DeepG4: {}".format(cmd))
    os.system(cmd)

    g4_scores = []
    with open(output_file, "r") as f:
        for line in f:
            line = line.strip()
            g4_scores.append(float(line))

    return np.array(g4_scores).reshape(-1, 1)


def _format_sequence(seq, max_seq_len, upstream_seq, downstream_seq, LEN_DEEPG4):
    assert len(seq) <= max_seq_len
    seq = upstream_seq + seq + downstream_seq
    seq = seq.upper()
    seq = seq.replace("U", "T")
    if len(seq) > LEN_DEEPG4:
        seq = seq[:LEN_DEEPG4]
    return seq