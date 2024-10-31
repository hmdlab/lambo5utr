import sys, os
from pathlib import Path
import numpy as np
from lambo5utr.tasks.mrl.prediction_mrl_CNN import predict_mrl_CNN
from lambo5utr.tasks.mrl.prediction_g4 import predict_g4
from lambo5utr.tasks.mrl.prediction_degradation import predict_in_vitro_degradation
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def main():
    upstream_seq = "GGGACATCGTAGAGAGTCGTACTTA"
    downstream_seq = "ATGGGCGAATTAAGTAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTAC"
    str_array = np.array(["AGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGAC",
                        "TCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGC"])

    project_root = Path(os.getcwd())#.parents[0]
    try:
        dirlist = os.listdir(project_root)
        assert (
            "python_scripts" in dirlist and 
            #"run_scripts" in dirlist and
            "lambo5utr" in dirlist and
            "hydra_config" in dirlist
        )
    except:
        raise AssertionError("Project root is not set as expected. Make sure to run the script under the ./run_scripts directory.")

    logger.info(
        "=======================================================\n" + 
        "Checking DeepG4 prediction\n" + 
        "=======================================================\n"
    )

    scores = predict_g4(
        str_array,
        max_seq_len=50,
        upstream_seq=upstream_seq,
        downstream_seq=downstream_seq,
        project_root=project_root,
    )

    logger.info(
        "=======================================================\n" + 
        "Check the predicted scores below.\n" +
        "The expected scores are 0.4325277 and 0.5933062\n" + 
        "=======================================================\n"
    )
    print(scores.shape)
    print(scores)

    logger.info(
        "=======================================================\n" + 
        "Checking Nullrecurrent prediction\n" + 
        "=======================================================\n"
    )

    scores = predict_in_vitro_degradation(
        str_array,
        max_seq_len=50,
        upstream_seq=upstream_seq,
        downstream_seq=downstream_seq,
        project_root=project_root,
    )

    logger.info(
        "=======================================================\n" + 
        "Check the predicted scores below.\n" +
        "The expected scores are 0.45495717 and 0.45235666\n" + 
        "=======================================================\n"
    )
    print(scores.shape)
    print(scores)

    logger.info(
        "=======================================================\n" + 
        " Checking Optimus5Prime prediction\n" + 
        "=======================================================\n"
    )

    scores = - predict_mrl_CNN(
        str_array,
        path_to_model="misc/retrained_main_MRL_model.hdf5",
        max_seq_len=50,
        project_root=project_root,
    ).reshape(-1, 1)

    logger.info(
        "=======================================================\n" + 
        "Check the predicted scores below.\n" +
        "The expected scores are 0.40430996 and 0.34016567 (normalized scores)\n" + 
        "=======================================================\n"
    )
    print(scores.shape)
    print(scores)

    return

if __name__ == "__main__":
    main()
    sys.exit()