import torch
import wandb
import math
import copy

from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm

from botorch.models import SingleTaskGP, KroneckerMultiTaskGP

from lambo5utr.utils import draw_bootstrap, to_tensor, weighted_resampling, batched_call
from lambo5utr import transforms as gfp_transforms
from lambo5utr.models.shared_elements import check_early_stopping
from lambo5utr.models.mlm import mlm_train_step, mlm_eval_epoch
from lambo5utr.models.lanmt import lanmt_eval_epoch, lanmt_train_step
from lambo5utr.models.lm_elements import LanguageModel

import logging

logger = logging.getLogger(__name__)

def gp_train_step(surrogate, optimizer, inputs, targets, mll):
    surrogate.zero_grad(set_to_none=True)
    surrogate.clear_cache()  # possibly unnecessary
    features = surrogate.get_features(
        inputs.to(surrogate.device), surrogate.bs, transform=False,
    ).double()

    dtype = features.dtype if (features.dtype is not torch.bool) else torch.float
    targets = surrogate.reshape_targets(targets).to(features.device, dtype=dtype)
    if isinstance(surrogate, (SingleTaskGP, KroneckerMultiTaskGP)):
        surrogate.set_train_data(features, targets, strict=False)
    output = surrogate.forward(features)
    loss = -mll(output, targets).mean()
    
    loss.backward()
    optimizer.step()
    loss.detach()

    del features
    torch.cuda.empty_cache()
    return loss


def fit_encoder_only(surrogate, optimizer, mll, train_loader, num_epochs):
    assert hasattr(surrogate, "encoder")
    surrogate.requires_grad_(False)
    surrogate.encoder.requires_grad_(True)
    for epoch_idx in range(num_epochs):
        surrogate.train()
        avg_loss = 0.0
        for inputs, targets in train_loader:
            loss = gp_train_step(surrogate, optimizer, inputs, targets, mll)
            avg_loss += loss.detach() / len(train_loader)
    return avg_loss.item()


def fit_gp_surrogate(
    surrogate,
    mll,
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    eval_bs=None,
    train_bs=None,
    shuffle_train=False,
    log_prefix="",
    encoder_obj="mll",
    resampling_temp=None,
):
    assert encoder_obj in ["mll", "mlm", "lanmt", None], "unsupported encoder objective"
    logger.info(
        f"{X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test"
        )

    if (
        surrogate.bootstrap_ratio is None and X_train.shape[0] >= surrogate.min_num_train
    ):
        pass
    else:
        X_train, Y_train = draw_bootstrap(
            X_train, Y_train, bootstrap_ratio=surrogate.bootstrap_ratio, min_samples=surrogate.min_num_train
        )

    # bias data towards 'good' examples
    if resampling_temp is not None:
        logger.info("---- resampling training and validation data ----")
        _, train_weights, train_idxs = weighted_resampling(-Y_train, k=resampling_temp)
        _, val_weights, val_idxs = weighted_resampling(-Y_val, k=resampling_temp)
        X_train, Y_train = X_train[train_idxs], Y_train[train_idxs]
        X_val, Y_val = X_val[val_idxs], Y_val[val_idxs]

    collate_fn = lambda x: gfp_transforms.padding_collate_fn(
        x, surrogate.tokenizer.padding_idx
    )
    train_bs = X_train.shape[0] if train_bs is None else train_bs
    train_dataset, val_dataset = surrogate._get_datasets(X_train, X_val, Y_train, Y_val)
    _, test_dataset = surrogate._get_datasets(X_train, X_test, Y_train, Y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=train_bs, shuffle=shuffle_train, collate_fn=collate_fn,
    )

    eval_bs = max(X_val.shape[0], X_test.shape[0]) if eval_bs is None else eval_bs
    val_loader = DataLoader(
        val_dataset, batch_size=eval_bs, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_bs, shuffle=False, collate_fn=collate_fn,
    )

    # prepare train targets to be passed to `surrogate.set_train_data`
    Y_train = to_tensor(Y_train, device=surrogate.device)
    Y_train = surrogate.reshape_targets(Y_train)
    Y_train = Y_train.to(dtype=list(surrogate.parameters())[0].dtype)

    if len(list(surrogate.encoder.parameters())) > 0:
        has_encoder = True
    else:
        logger.info("---- surrogate has no encoder ----")
        has_encoder = False

    logger.info("---- preparing checkpoint ----")
    # evaluate starting model
    surrogate.eval()
    surrogate.requires_grad_(False)
    surrogate.set_train_data(X_train, Y_train, strict=False)
    start_metrics = surrogate.evaluate(val_loader, split='val')
    start_metrics.update(surrogate.evaluate(test_loader, split='test'))
    start_metrics['epoch'] = 0

    if has_encoder and encoder_obj == 'mlm':
        start_metrics.update(
            mlm_eval_epoch(
                surrogate.encoder, val_loader, surrogate.encoder.mask_ratio, split='val'
            )
        )
        start_metrics.update(
            mlm_eval_epoch(
                surrogate.encoder, test_loader, surrogate.encoder.mask_ratio, split='test'
            )
        )
    if has_encoder and encoder_obj == 'lanmt':
        start_metrics.update(
            lanmt_eval_epoch(surrogate.encoder.model, val_loader, split='val')
        )
        start_metrics.update(
            lanmt_eval_epoch(surrogate.encoder.model, test_loader, split='test')
        )

    select_crit_key = "val_nll"
    best_score = start_metrics[select_crit_key]
    best_score_epoch = 0
    surrogate.cpu()  # avoid storing two copies of the weights on GPU
    best_weights = copy.deepcopy(surrogate.state_dict())
    surrogate.to(surrogate.device)
    logger.info(f"starting val NLL: {best_score:.4f}")
    if any([isinstance(module, _BatchNorm) for module in surrogate.encoder.modules()]):
        logger.info("---- initializing encoder normalization buffers ----")
        num_warmup_epochs = 8
        surrogate.train()
        surrogate.requires_grad_(False)
        for epoch in range(num_warmup_epochs):
            for inputs, _ in train_loader:
                _ = surrogate.get_features(inputs.to(surrogate.device), surrogate.bs, transform=False)

    mll.to(surrogate.device)
    if hasattr(mll, "num_data"):
        mll.num_data = len(train_loader.dataset)

    stop_crit_key = "train_loss"
    best_loss, best_loss_epoch = None, 0
    stop = False

    gp_optimizer = torch.optim.Adam(surrogate.param_groups)
    gp_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gp_optimizer, patience=math.ceil(surrogate.patience / 2.0), threshold=1e-3
    )

    records = [start_metrics]
    logger.info("---- fitting all params ----")
    for epoch_idx in range(surrogate.num_epochs):
        metrics = {}

        # train encoder through supervised MLL objective
        if has_encoder and encoder_obj == "mll":
            enc_sup_loss = fit_encoder_only(
                surrogate, gp_optimizer, mll, train_loader, num_epochs=1
            )
        else:
            enc_sup_loss = 0.0

        avg_train_loss = enc_sup_loss
        surrogate.train()
        for inputs, targets in train_loader:

            # train encoder through unsupervised MLM objective
            if isinstance(surrogate.encoder, LanguageModel) and encoder_obj == "mlm":
                surrogate.encoder.requires_grad_(True)
                mlm_loss, _, _ = mlm_train_step(
                    surrogate.encoder,
                    gp_optimizer,
                    inputs,
                    surrogate.encoder.mask_ratio,
                    loss_scale=1.0,
                )
            elif (
                isinstance(surrogate.encoder, LanguageModel) and encoder_obj == "lanmt"
            ):
                surrogate.encoder.requires_grad_(True)
                mlm_loss, _, _ = lanmt_train_step(
                    surrogate.encoder.model, gp_optimizer, inputs, loss_scale=1.0
                )
            else:
                mlm_loss = torch.zeros(1, device=surrogate.device)
            # train all params through supervised MLL objective
            surrogate.requires_grad_(True)
            gp_loss = gp_train_step(surrogate, gp_optimizer, inputs, targets, mll)
            avg_train_loss += (mlm_loss.detach() + gp_loss.detach()) / len(train_loader)
        gp_lr_sched.step(avg_train_loss)
        metrics.update(
            {
                "epoch": epoch_idx + 1,
                "train_loss": avg_train_loss.item(),
            }
        )

        if (epoch_idx + 1) % surrogate.eval_period == 0:
            surrogate.requires_grad_(False)
            # update train features, use unaugmented train data for evaluation
            surrogate.eval()
            surrogate.set_train_data(X_train, Y_train, strict=False)

            metrics.update(surrogate.evaluate(val_loader, split="val"))
            metrics.update(surrogate.evaluate(test_loader, split="test"))
            if has_encoder and encoder_obj == "mlm":
                metrics.update(
                    mlm_eval_epoch(
                        surrogate.encoder,
                        val_loader,
                        surrogate.encoder.mask_ratio,
                        split="val"
                    )
                )
                metrics.update(
                    mlm_eval_epoch(
                        surrogate.encoder,
                        test_loader,
                        surrogate.encoder.mask_ratio,
                        split="test"
                    )
                )
            elif has_encoder and encoder_obj == "lanmt":
                metrics.update(
                    lanmt_eval_epoch(
                        surrogate.encoder.model,
                        val_loader,
                        split="val"
                    )
                )
                metrics.update(
                    lanmt_eval_epoch(
                        surrogate.encoder.model,
                        test_loader,
                        split="test"
                    )
                )

        # use validation NLL for model selection
        select_crit = metrics.get(select_crit_key, None)
        if surrogate.early_stopping and select_crit is not None:
            assert (
                surrogate.holdout_ratio > 0.0
            ), "Need validation data for early stopping"
            best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                model=surrogate,
                best_score=best_score,
                best_epoch=best_score_epoch,
                best_weights=best_weights,
                curr_score=select_crit,
                curr_epoch=epoch_idx + 1,
                patience=surrogate.patience,
                save_weights=True,
            )
        metrics.update(dict(best_score=best_score, best_epoch=best_score_epoch))

        # use train loss to determine convergence
        stop_crit = metrics.get(stop_crit_key, None)
        if stop_crit is not None:
            best_loss, best_loss_epoch, _, stop = check_early_stopping(
                model=surrogate,
                best_score=best_loss,
                best_epoch=best_loss_epoch,
                best_weights=None,
                curr_score=stop_crit,
                curr_epoch=epoch_idx + 1,
                patience=surrogate.patience,
                save_weights=False,
            )
        metrics.update(dict(best_loss=best_loss, best_loss_epoch=best_loss_epoch))

        records.append(metrics)
        if len(log_prefix) > 0:
            metrics = {"/".join((log_prefix, key)): val for key, val in metrics.items()}
        try:
            wandb.log(metrics)
        except Exception:
            pass

        if stop:
            break

    if surrogate.early_stopping:
        logger.info(f"---- loading checkpoint from epoch {best_score_epoch} ----")
        surrogate.load_state_dict(best_weights)
    surrogate.requires_grad_(False)
    surrogate.train()  # clear caches
    surrogate.clear_cache()
    surrogate.eval()
    surrogate.set_train_data(X_train, Y_train, strict=False)

    return records