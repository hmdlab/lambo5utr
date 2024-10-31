import numpy as np
import gc
from tensorflow import keras
import tensorflow as tf
import logging
import os
import sys
import argparse

logger = logging.getLogger(__name__)

def _one_hot_encode(str_array, max_seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides. 
    nuc_d = {"a":[1,0,0,0],
             "c":[0,1,0,0],
             "g":[0,0,1,0],
             "t":[0,0,0,1],
    }
    
    vectors = np.empty([str_array.shape[0],max_seq_len,4])
    for i, seq in enumerate(str_array): 
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

def main(args):
    path_to_model = args.path_model
    # "/home/keisuke-yamada/lambo5utr/notebooks/20221030_implement_CNNmrlpred/retrained_main_MRL_model.hdf5"
    logger.info("loading MRL Predictor (CNN) from {}".format(path_to_model))
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(
                "{} memory growth: {}".format(
                    device, tf.config.experimental.get_memory_growth(device)
                )
            )
    else:
        print("Not enough GPU hardware devices available")
    model = keras.models.load_model(path_to_model)
    logger.info("finished loading MRL Predictor (CNN)")

    input_seq = []
    with open(args.input, mode="r") as f:
        for line in f:
            input_seq.append(line.strip())
    input_seq = np.array(input_seq)

    encoded_seq = _one_hot_encode(input_seq)
    preds = model(encoded_seq).numpy().reshape(-1)

    with open(args.output, mode="w") as f:
        preds = list(preds.astype(str))
        lines = "\n".join(preds)
        f.write(lines)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="input_file")
    parser.add_argument("--output", type=str, default=None, help="output_file")
    parser.add_argument("--path_model", type=str, default=None, help="path to the saved model")
    args = parser.parse_args()
    main(args)
    sys.exit()

