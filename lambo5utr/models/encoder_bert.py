import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from lambo5utr.models.masked_layers import Apply, mResidualBlock
from lambo5utr.models.shared_elements import BaseEncoder, Expression
from lambo5utr.models.lm_elements import LamboBaseLM, FunctionHead, PoolingHead, LengthHead, LengthTransform

class BertEncoderLmheadForLambo(BaseEncoder, BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, tokenizer, **kwargs):
        BertPreTrainedModel.__init__(self, config)
        BaseEncoder.__init__(self, tokenizer, config=config, **kwargs)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.out_dim = config.out_dim
        self.embed_dim = config.hidden_size
        self.latent_dim = config.hidden_size
        self.max_len_delta = config.max_len_delta

        self.decoder = nn.Sequential(
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
            mResidualBlock(self.latent_dim * 2, self.embed_dim, config.kernel_size, config.layernorm, config.p),
            mResidualBlock(self.embed_dim, self.embed_dim, config.kernel_size, config.layernorm, config.p),
            mResidualBlock(self.embed_dim, self.embed_dim, config.kernel_size, config.layernorm, config.p),
            # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
            # Apply(nn.Dropout(p=p)),
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        )
        self.length_transform = LengthTransform()
        self.function_head = FunctionHead(self.latent_dim, self.out_dim, config.kernel_size, config.layernorm, config.p, None, type='conv')
        self.length_head = LengthHead(self.latent_dim, self.max_len_delta)
        # this must be the name "cls" to import pretrained parameters
        # this is equivalent to lm_head in LanguageModel
        self.lm_head = BertOnlyMLMHead(config)

    def enc_tok_features(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_mask = (src_tok_idxs != self.tokenizer.padding_idx).float()
        src_tok_features = self.bert(input_ids = src_tok_idxs, attention_mask=src_mask, return_dict=False)[0]
        # src_tok_features.shape = batch_size * max_len * latent_dim
        return src_tok_features, src_mask

    def dec_tok_features(self, src_tok_features, src_mask, lat_tok_features=None, tgt_lens=None):
        # internal features from function head
        if lat_tok_features is None:
            lat_tok_features, _ = self.function_head(
                src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
            )

        len_delta_logits = self.length_head(src_tok_features, src_mask)
        # predict target seq length if unknown
        if tgt_lens is None:
            src_lens = src_mask.float().sum(-1)
            tgt_lens = self.length_head.sample(src_lens, len_delta_logits)

        tgt_tok_features, tgt_mask = self.length_transform(
            src_tok_features=torch.cat([src_tok_features, lat_tok_features], dim=-1),
            src_mask=src_mask,
            tgt_lens=tgt_lens
        )

        tgt_tok_features, _ = self.decoder((tgt_tok_features, tgt_mask))

        return lat_tok_features, tgt_tok_features, tgt_mask, len_delta_logits

    def tgt_tok_logits(self, tgt_tok_features):
        reshaped_features = tgt_tok_features.flatten(end_dim=-2)
        logits = self.lm_head(reshaped_features)
        logits = logits.view(*tgt_tok_features.shape[:-1], -1)
        return logits

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.sep_idx)
        _, pooled_features = self.function_head(src_tok_features, src_mask, pooling_mask)
        return pooled_features
    
    def param_groups(self, lr, weight_decay=0., pretrain_lr=None):
        shared_group = dict(params=[], lr=lr, weight_decay=weight_decay, betas=(0., 1e-2))
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)
        pretrained_group = dict(params=[], lr=pretrain_lr, weight_decay=weight_decay)

        if pretrain_lr is None:
            shared_names = ['bert', 'function_head']
        else:
            shared_names = ['function_head']
            pretrained_names = ['bert']
            # pretrained_names = ['bert']
        for p_name, param in self.named_parameters():
            prefix = p_name.split('.')[0]
            if prefix in shared_names:
                shared_group['params'].append(param)
            elif prefix in pretrained_names:
                pretrained_group['params'].append(param)
            else:
                other_group['params'].append(param)

        if pretrain_lr == 0:
            return shared_group, other_group
        else:
            return shared_group, pretrained_group, other_group

class DnaBertLmheadForLambo(LamboBaseLM):
    def __init__(self, pretrained_path, batch_size, num_epochs, patience, max_shift, mask_ratio, 
                fromscratch=False, tokenizer=None, **kwargs):
        LamboBaseLM.__init__(self, **kwargs)
        self.model_config = BertConfig
        self.model_config = self.model_config.from_pretrained(
                                pretrained_path,
                                )
        
        additional_configs = ['out_dim', 'kernel_size', 'p', 
                            'layernorm', 'max_len_delta',
                            'max_seq_len', 'max_len', 'num_heads']
        for add_cfg in additional_configs:
            if add_cfg in kwargs:
                setattr(self.model_config, add_cfg, kwargs[add_cfg])
            else:
                setattr(self.model_config, add_cfg, None)
        
        # print(self.model_config)

        self.model = BertEncoderLmheadForLambo
        if fromscratch:
            self.model = self.model(config=self.model_config,
                                    tokenizer=tokenizer,
                                    max_seq_len=self.model_config.max_seq_len,
                                    max_len=self.model_config.max_len,
                                    ).to(self.device)
        else:
            self.model = self.model.from_pretrained(
                                    pretrained_path,
                                    config=self.model_config,
                                    tokenizer=tokenizer,
                                    max_seq_len=self.model_config.max_seq_len,
                                    max_len=self.model_config.max_len,
                                    ).to(self.device)
        
        # print(self.model)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.max_shift = max_shift
        self.tokenizer = self.model.tokenizer
        self.mask_ratio = mask_ratio

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            tok_idxs = self.str_to_tokens(inputs)
        else:
            tok_idxs = inputs
        return self.model(tok_idxs)

    def pool_features(self, src_tok_features, src_mask):
        lat_tok_features, pooled_features = self.model.function_head(
            src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
        )
        return lat_tok_features, pooled_features

    def get_token_idx(self, token):
        return self.model.tokenizer.convert_token_to_id(token)

    def get_token(self, idx):
        return self.model.tokenizer.convert_id_to_token(idx)

    def get_token_features(self, src_tok_idxs):
        src_tok_features, src_mask = self.model.enc_tok_features(src_tok_idxs)
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(src_tok_features, src_mask, lat_tok_features=None)
        return tgt_tok_logits, tgt_mask

    def logits_from_features(self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None):
        lat_tok_features, tgt_tok_features, tgt_mask, _ = self.model.dec_tok_features(
            src_tok_features, src_mask, lat_tok_features, tgt_lens
        )
        tgt_tok_logits = self.model.tgt_tok_logits(tgt_tok_features)
        return tgt_tok_logits, tgt_mask

    def sample_tgt_tok_idxs(self, tgt_tok_logits, tgt_mask, temp=1.):
        batch_size, num_tokens = tgt_mask.shape
        tgt_lens = tgt_mask.float().sum(-1).long()
        tgt_tok_logits /= temp

        # don't sample special tokens
        non_viable_idxs = np.array(self.tokenizer.special_idxs)[None, None, :]
        np.put_along_axis(tgt_tok_logits, non_viable_idxs, -1e10, axis=-1)

        tgt_tok_idxs = torch.full(tgt_mask.size(), self.tokenizer.padding_idx)
        tgt_tok_idxs = tgt_tok_idxs.to(tgt_mask).long()
        tok_dist = torch.distributions.Categorical(logits=tgt_tok_logits)
        sample_tok_idxs = tok_dist.sample()

        tgt_tok_idxs += tgt_mask * sample_tok_idxs

        tgt_tok_idxs[:, 0] = self.tokenizer.cls_idx
        tgt_tok_idxs[torch.arange(batch_size), tgt_lens - 1] = self.tokenizer.sep_idx

        logit_entropy = -(
            F.softmax(tgt_tok_logits, dim=-1) * F.log_softmax(tgt_tok_logits, dim=-1)
        ).sum(-1)
        logit_entropy *= tgt_mask.float()
        logit_entropy = logit_entropy.sum() / tgt_mask.float().sum()

        return tgt_tok_idxs, logit_entropy

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)

class BertEncoderForLambo(BaseEncoder, BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, tokenizer, **kwargs):
        BertPreTrainedModel.__init__(self, config)
        BaseEncoder.__init__(self, tokenizer, config=config, **kwargs)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.out_dim = config.out_dim
        self.embed_dim = config.hidden_size
        self.latent_dim = config.hidden_size
        self.max_len_delta = config.max_len_delta

        self.decoder = nn.Sequential(
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
            mResidualBlock(self.latent_dim * 2, self.embed_dim, config.kernel_size, config.layernorm, config.p),
            mResidualBlock(self.embed_dim, self.embed_dim, config.kernel_size, config.layernorm, config.p),
            mResidualBlock(self.embed_dim, self.embed_dim, config.kernel_size, config.layernorm, config.p),
            # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
            # Apply(nn.Dropout(p=p)),
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        )
        self.length_transform = LengthTransform()
        self.function_head = FunctionHead(self.latent_dim, self.out_dim, config.kernel_size, config.layernorm, config.p, None, type='conv')
        
        self.length_head = LengthHead(self.latent_dim, self.max_len_delta)
        # this must be the name "cls" to import pretrained parameters
        # this is equivalent to lm_head in LanguageModel
        self.cls = BertOnlyMLMHead(config)
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def enc_tok_features(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_mask = (src_tok_idxs != self.tokenizer.padding_idx).float()
        src_tok_features = self.bert(input_ids = src_tok_idxs, attention_mask=src_mask, return_dict=False)[0]
        # src_tok_features.shape = batch_size * max_len * latent_dim
        return src_tok_features, src_mask

    def dec_tok_features(self, src_tok_features, src_mask, lat_tok_features=None, tgt_lens=None):
        # internal features from function head
        if lat_tok_features is None:
            lat_tok_features, _ = self.function_head(
                src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
            )

        len_delta_logits = self.length_head(src_tok_features, src_mask)
        # predict target seq length if unknown
        if tgt_lens is None:
            src_lens = src_mask.float().sum(-1)
            tgt_lens = self.length_head.sample(src_lens, len_delta_logits)

        tgt_tok_features, tgt_mask = self.length_transform(
            src_tok_features=torch.cat([src_tok_features, lat_tok_features], dim=-1),
            src_mask=src_mask,
            tgt_lens=tgt_lens
        )

        tgt_tok_features, _ = self.decoder((tgt_tok_features, tgt_mask))

        return lat_tok_features, tgt_tok_features, tgt_mask, len_delta_logits

    def tgt_tok_logits(self, tgt_tok_features):
        reshaped_features = tgt_tok_features.flatten(end_dim=-2)
        logits = self.cls(reshaped_features)
        logits = logits.view(*tgt_tok_features.shape[:-1], -1)
        return logits

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.sep_idx)
        _, pooled_features = self.function_head(src_tok_features, src_mask, pooling_mask)
        return pooled_features
    
    def param_groups(self, lr, weight_decay=0., pretrain_lr=None):
        shared_group = dict(params=[], lr=lr, weight_decay=weight_decay, betas=(0., 1e-2))
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)
        pretrained_group = dict(params=[], lr=pretrain_lr, weight_decay=weight_decay)

        if pretrain_lr is None:
            shared_names = ['bert', 'function_head']
        else:
            shared_names = ['function_head']
            pretrained_names = ['bert']
            # pretrained_names = ['bert']
        for p_name, param in self.named_parameters():
            prefix = p_name.split('.')[0]
            if prefix in shared_names:
                shared_group['params'].append(param)
            elif prefix in pretrained_names:
                pretrained_group['params'].append(param)
            else:
                other_group['params'].append(param)

        if pretrain_lr == 0:
            return shared_group, other_group
        else:
            return shared_group, pretrained_group, other_group

class DnaBertForLambo(LamboBaseLM):
    def __init__(self, pretrained_path, batch_size, num_epochs, patience, max_shift, mask_ratio, 
                fromscratch=False, tokenizer=None, **kwargs):
        LamboBaseLM.__init__(self, **kwargs)
        self.model_config = BertConfig
        self.model_config = self.model_config.from_pretrained(
                                pretrained_path,
                                )
        
        additional_configs = ['out_dim', 'kernel_size', 'p', 
                            'layernorm', 'max_len_delta',
                            'max_seq_len', 'max_len', 'num_heads']
        for add_cfg in additional_configs:
            if add_cfg in kwargs:
                setattr(self.model_config, add_cfg, kwargs[add_cfg])
            else:
                setattr(self.model_config, add_cfg, None)
        
        # print(self.model_config)

        self.model = BertEncoderForLambo
        if fromscratch:
            self.model = self.model(config=self.model_config,
                                    tokenizer=tokenizer,
                                    max_seq_len=self.model_config.max_seq_len,
                                    max_len=self.model_config.max_len,
                                    ).to(self.device)
        else:
            self.model = self.model.from_pretrained(
                                    pretrained_path,
                                    config=self.model_config,
                                    tokenizer=tokenizer,
                                    max_seq_len=self.model_config.max_seq_len,
                                    max_len=self.model_config.max_len,
                                    ).to(self.device)
        
        # print(self.model)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.max_shift = max_shift
        self.tokenizer = self.model.tokenizer
        self.mask_ratio = mask_ratio

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            tok_idxs = self.str_to_tokens(inputs)
        else:
            tok_idxs = inputs
        return self.model(tok_idxs)

    def pool_features(self, src_tok_features, src_mask):
        lat_tok_features, pooled_features = self.model.function_head(
            src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
        )
        return lat_tok_features, pooled_features

    def get_token_idx(self, token):
        return self.model.tokenizer.convert_token_to_id(token)

    def get_token(self, idx):
        return self.model.tokenizer.convert_id_to_token(idx)

    def get_token_features(self, src_tok_idxs):
        src_tok_features, src_mask = self.model.enc_tok_features(src_tok_idxs)
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(src_tok_features, src_mask, lat_tok_features=None)
        return tgt_tok_logits, tgt_mask

    def logits_from_features(self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None):
        lat_tok_features, tgt_tok_features, tgt_mask, _ = self.model.dec_tok_features(
            src_tok_features, src_mask, lat_tok_features, tgt_lens
        )
        tgt_tok_logits = self.model.tgt_tok_logits(tgt_tok_features)
        return tgt_tok_logits, tgt_mask

    def sample_tgt_tok_idxs(self, tgt_tok_logits, tgt_mask, temp=1.):
        batch_size, num_tokens = tgt_mask.shape
        tgt_lens = tgt_mask.float().sum(-1).long()
        tgt_tok_logits /= temp

        # don't sample special tokens
        non_viable_idxs = np.array(self.tokenizer.special_idxs)[None, None, :]
        np.put_along_axis(tgt_tok_logits, non_viable_idxs, -1e10, axis=-1)

        tgt_tok_idxs = torch.full(tgt_mask.size(), self.tokenizer.padding_idx)
        tgt_tok_idxs = tgt_tok_idxs.to(tgt_mask).long()
        tok_dist = torch.distributions.Categorical(logits=tgt_tok_logits)
        sample_tok_idxs = tok_dist.sample()

        tgt_tok_idxs += tgt_mask * sample_tok_idxs

        tgt_tok_idxs[:, 0] = self.tokenizer.cls_idx
        tgt_tok_idxs[torch.arange(batch_size), tgt_lens - 1] = self.tokenizer.sep_idx

        logit_entropy = -(
            F.softmax(tgt_tok_logits, dim=-1) * F.log_softmax(tgt_tok_logits, dim=-1)
        ).sum(-1)
        logit_entropy *= tgt_mask.float()
        logit_entropy = logit_entropy.sum() / tgt_mask.float().sum()

        return tgt_tok_idxs, logit_entropy

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)

class BertOnlyEncoderForLambo(BaseEncoder, BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, tokenizer, **kwargs):
        BertPreTrainedModel.__init__(self, config)
        BaseEncoder.__init__(self, tokenizer, config=config, **kwargs)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.out_dim = config.out_dim
        self.embed_dim = config.hidden_size
        self.latent_dim = config.hidden_size
        self.max_len_delta = config.max_len_delta

        self.function_head = PoolingHead(self.latent_dim, self.out_dim)

        # this must be the name "cls" to import pretrained parameters
        # this is equivalent to lm_head in LanguageModel
        self.cls = BertOnlyMLMHead(config)
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def enc_tok_features(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_mask = (src_tok_idxs != self.tokenizer.padding_idx).float()
        src_tok_features = self.bert(input_ids = src_tok_idxs, attention_mask=src_mask, return_dict=False)[0]
        # src_tok_features.shape = batch_size * max_len * latent_dim
        return src_tok_features, src_mask

    """
    def dec_tok_features(self, src_tok_features, src_mask, lat_tok_features=None, tgt_lens=None):

        tgt_tok_features = self.cls(src_tok_features)

        return src_tok_features, tgt_tok_features, src_mask, None
    """

    def tgt_tok_logits(self, tgt_tok_features):
        # reshaped_features = tgt_tok_features.flatten(end_dim=-2)
        logits = self.cls(tgt_tok_features)
        logits = logits.view(*tgt_tok_features.shape[:-1], -1)
        return logits

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, :self.max_len + 1]
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.sep_idx)
        _, pooled_features = self.function_head(src_tok_features, src_mask, pooling_mask)
        return pooled_features
    
    def param_groups(self, lr, weight_decay=0., pretrain_lr=None):
        shared_group = dict(params=[], lr=lr, weight_decay=weight_decay, betas=(0., 1e-2))
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)
        pretrained_group = dict(params=[], lr=pretrain_lr, weight_decay=weight_decay)

        if pretrain_lr is None:
            shared_names = ['bert', 'function_head']
        else:
            shared_names = ['function_head', 'cls']
            pretrained_names = ['bert']
            # pretrained_names = ['bert']
        for p_name, param in self.named_parameters():
            prefix = p_name.split('.')[0]
            if prefix in shared_names:
                shared_group['params'].append(param)
            elif prefix in pretrained_names:
                pretrained_group['params'].append(param)
            else:
                other_group['params'].append(param)

        if pretrain_lr == 0 or pretrain_lr is None:
            return shared_group, other_group
        else:
            return shared_group, pretrained_group, other_group

class DnaBertOnlyForLambo(LamboBaseLM):
    def __init__(self, pretrained_path, batch_size, num_epochs, patience, max_shift, mask_ratio, 
                fromscratch=False, tokenizer=None, **kwargs):
        LamboBaseLM.__init__(self, **kwargs)
        self.model_config = BertConfig
        self.model_config = self.model_config.from_pretrained(
                                pretrained_path,
                                )
        
        additional_configs = ['out_dim', 'kernel_size', 'p', 
                            'layernorm', 'max_len_delta',
                            'max_seq_len', 'max_len',]
        for add_cfg in additional_configs:
            if add_cfg in kwargs:
                setattr(self.model_config, add_cfg, kwargs[add_cfg])
            else:
                setattr(self.model_config, add_cfg, None)
        
        # print(self.model_config)

        self.model = BertOnlyEncoderForLambo
        if fromscratch:
            self.model = self.model(config=self.model_config,
                                    tokenizer=tokenizer,
                                    max_seq_len=self.model_config.max_seq_len,
                                    max_len=self.model_config.max_len,
                                    ).to(self.device)
        else:
            self.model = self.model.from_pretrained(
                                    pretrained_path,
                                    config=self.model_config,
                                    tokenizer=tokenizer,
                                    max_seq_len=self.model_config.max_seq_len,
                                    max_len=self.model_config.max_len,
                                    ).to(self.device)
        
        # print(self.model)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.max_shift = max_shift
        self.tokenizer = self.model.tokenizer
        self.mask_ratio = mask_ratio

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            tok_idxs = self.str_to_tokens(inputs)
        else:
            tok_idxs = inputs
        return self.model(tok_idxs)

    def pool_features(self, src_tok_features, src_mask):
        lat_tok_features, pooled_features = self.model.function_head(
            src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
        )
        return lat_tok_features, pooled_features

    def get_token_idx(self, token):
        return self.model.tokenizer.convert_token_to_id(token)

    def get_token(self, idx):
        return self.model.tokenizer.convert_id_to_token(idx)

    def get_token_features(self, src_tok_idxs):
        src_tok_features, src_mask = self.model.enc_tok_features(src_tok_idxs)
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(src_tok_features, src_mask, lat_tok_features=None)
        return tgt_tok_logits, tgt_mask

    def logits_from_features(self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None):
        # lat_tok_features, tgt_tok_features, tgt_mask, _ = self.model.dec_tok_features(
        #    src_tok_features, src_mask, lat_tok_features, tgt_lens
        #)
        tgt_tok_logits = self.model.tgt_tok_logits(src_tok_features)
        return tgt_tok_logits, src_mask

    def sample_tgt_tok_idxs(self, tgt_tok_logits, tgt_mask, temp=1.):
        batch_size, num_tokens = tgt_mask.shape
        tgt_lens = tgt_mask.float().sum(-1).long()
        tgt_tok_logits /= temp

        # don't sample special tokens
        non_viable_idxs = np.array(self.tokenizer.special_idxs)[None, None, :]
        np.put_along_axis(tgt_tok_logits, non_viable_idxs, -1e10, axis=-1)

        tgt_tok_idxs = torch.full(tgt_mask.size(), self.tokenizer.padding_idx)
        tgt_tok_idxs = tgt_tok_idxs.to(tgt_mask).long()
        tok_dist = torch.distributions.Categorical(logits=tgt_tok_logits)
        sample_tok_idxs = tok_dist.sample()

        tgt_tok_idxs += tgt_mask * sample_tok_idxs

        tgt_tok_idxs[:, 0] = self.tokenizer.cls_idx
        tgt_tok_idxs[torch.arange(batch_size), tgt_lens - 1] = self.tokenizer.sep_idx

        logit_entropy = -(
            F.softmax(tgt_tok_logits, dim=-1) * F.log_softmax(tgt_tok_logits, dim=-1)
        ).sum(-1)
        logit_entropy *= tgt_mask.float()
        logit_entropy = logit_entropy.sum() / tgt_mask.float().sum()

        return tgt_tok_idxs, logit_entropy

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)