import hydra
import warnings
import random
import logging
import os
import re
from pathlib import Path
import sys
from upcycle.logging.analysis import flatten_config
from gpytorch.settings import max_cholesky_size
from omegaconf import OmegaConf, DictConfig
import randomname, random
import torch
import numpy as np

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms = True

def startup(hydra_cfg):
    trial_id = hydra_cfg.trial_id
    if hydra_cfg.job_name is None:
        hydra_cfg.job_name = "_".join(
            randomname.get_name().lower().split("-") + [str(trial_id)]
        )
    hydra_cfg.seed = (
        random.randint(0, 100000) if hydra_cfg.seed is None else hydra_cfg.seed
    )
    set_all_seeds(hydra_cfg.seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(
        hydra_cfg, resolve=True
    )  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    # print(OmegaConf.to_yaml(hydra_cfg))
    with open("hydra_config.txt", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))
    print(f"GPU available: {torch.cuda.is_available()}")
    return hydra_cfg, logger

@hydra.main(
    config_path="../hydra_config", config_name="black_box_opt", version_base="1.2"
)
def main(config):
    # setup
    random.seed(
        None
    )  # make sure random seed resets between multirun jobs for random job-name generation
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}
    if hasattr(config["task"], "given_filename"):
        name = (
            config["task"]["given_filename"].split("_")[-1]
            + "_scratch"
            + "-seed"
            + str(config["seed"])
        )
    else:
        name = "test"
    config["job_name"] = name
    import wandb

    wandb.init(
        project="lambo",
        name=name,
        config=log_config,
        mode=config.wandb_mode,
        group=config.exp_name,
    )
    
    config, _ = startup(config)  # random seed is fixed here

    # changing the Hydra run dir will break the project_root setting.
    project_root = Path(os.getcwd())# .parents[0]
    try:
        dirlist = os.listdir(project_root)
        assert (
            "python_scripts" in dirlist and 
            "run_scripts" in dirlist and
            "lambo5utr" in dirlist and
            "hydra_config" in dirlist
        )
    except:
        raise AssertionError("Project root is not set as expected. Make sure to run the script under the ./run_scripts directory.")

    logger = logging.getLogger("lambo5utr")
    file_handler = logging.FileHandler(
        filename=os.path.join(
            os.path.join(project_root),
            config.log_dir,
            config.job_name,
            config.timestamp,
            "output.log",
        )
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # catch BioPython warnings
        try:
            # create initial candidates, dataset
            tokenizer = hydra.utils.instantiate(config.tokenizer)
            if hasattr(config.tokenizer, "vocab_file"):
                print(
                    "initializing main tokenizer using pretrained file at {}".format(
                        config.tokenizer.vocab_file
                    )
                )
                tokenizer = tokenizer.from_pretrained(config.tokenizer.vocab_file)

            exp_name = config.exp_name + "-" + name
            config.task.exp_name = exp_name
            bb_task = hydra.utils.instantiate(
                config.task, tokenizer=tokenizer, candidate_pool=[]
            )

            base_candidates, base_targets, all_seqs, all_targets = bb_task.task_setup(
                config, project_root=project_root
            )

            # optimizer
            max_chol_sz = config.surrogate.get("max_cholesky_size", int(1e5))
            with max_cholesky_size(max_chol_sz):
                optimizer = hydra.utils.instantiate(
                    config.optimizer,
                    bb_task=config.task,
                    surrogate=config.surrogate,
                    acquisition=config.acquisition,
                    encoder=config.encoder,
                    tokenizer=tokenizer,
                )
                metrics = optimizer.optimize(
                    base_candidates,
                    base_targets,
                    all_seqs,
                    all_targets,
                    log_prefix=config.task.log_prefix,
                )
            if len(metrics) == 2:
                metrics = metrics[0]
            metrics = {
                key.split("/")[-1]: val for key, val in metrics.items()
            }  # strip prefix
            ret_val = metrics["hypervol_rel"]

        except Exception as err:
            logging.exception(err)
            ret_val = float("NaN")

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val


if __name__ == "__main__":
    main()
    sys.exit()
