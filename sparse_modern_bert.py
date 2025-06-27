from typing import Optional, Tuple

import torch
from transformers import ModernBertConfig, ModernBertModel, ModernBertForMaskedLM
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertAttention,
    ModernBertForSequenceClassification,
    MODERNBERT_ATTENTION_FUNCTION,
    apply_rotary_pos_emb
)

# pip install entmax
from entmax import entmax_bisect, entmax15, sparsemax

# pip install adasplash
from adasplash import adasplash, adasplash_no_block_mask, triton_entmax

def adasplash_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    alpha: float,
    pre_niter: int = 5,
    post_niter: int = 5,
    output_attentions: Optional[bool] = False,
    **_kwargs,
):
    assert not output_attentions, "Output attentions not supported"
    assert local_attention == (-1, -1), "Local attention not supported"

    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # ensure that the input is contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # compute attention
    valid_mask = attention_mask[:, 0, 0] == 0
    varlen = valid_mask.sum(-1).long().contiguous()
    max_seqlen = varlen.max().item()

    assert varlen.min().item() >= 1, "Some sequences in the batch are empty"

    # try also with `adasplash()` for more efficient training on long context lengths
    attn_output = adasplash_no_block_mask(
        query, 
        key, 
        value,  
        alpha=alpha,
        niter=pre_niter,
        is_causal=False,
        varlen=varlen
    )

    # ensure that the output is shaped correctly
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, -1, dim)

    return (attn_output,)

def entmax_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    alpha: float,
    pre_niter: int = 5,
    post_niter: int = 5,
    output_attentions: Optional[bool] = False,
    **_kwargs,
):
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    if alpha == 1.0:
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    else:
        attn_weights = triton_entmax(attn_weights, alpha=alpha, n_iter=pre_niter, fast_math=True)

    attn_weights = attn_weights.to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)



class CustomModernBertAttention(ModernBertAttention):
    def __init__(self, config, layer_id=None):
        super().__init__(config, layer_id=layer_id)
        self.alpha = config.initial_alpha if hasattr(config, 'initial_alpha') else 2.0
        self.pre_iter = config.pre_iter if hasattr(config, 'pre_iter') else 5
        self.post_iter = config.post_iter if hasattr(config, 'post_iter') else 5
        self.use_triton_entmax = config.use_triton_entmax if hasattr(config, 'use_triton_entmax') else False
        print('>>>>>>>> LAYER ID:', layer_id)
        print('>>>>>>>> LOCAL ATTENTION:', self.local_attention)
        print('>>>>>>>> ENTMAX ALPHA:', self.alpha)
        print('>>>>>>>> USE TRITON ENTMAX:', self.use_triton_entmax)
        print('>>>>>>>> PRE ITER:', self.pre_iter)
        print('>>>>>>>> POST ITER:', self.post_iter)
        # self.register_buffer("sparsity_per_head", torch.zeros(1, config.num_attention_heads))  # Buffer for sparsity
        self.sparsity_per_head = torch.zeros(1, config.num_attention_heads)
        # self.register_buffer("n_tokens", torch.tensor(0))  # Buffer for the number of tokens
        self.n_tokens = torch.tensor(0)
        # self.opt_entmax_bisect_fn = torch.compile(entmax_bisect)  # Compile entmax function for faster execution
        self.opt_entmax_bisect_fn = entmax_bisect

    # do not try to load sparsity_per_head and n_tokens when loading from a checkpoint
    def load_state_dict(self, state_dict, **kwargs):
        if 'sparsity_per_head' in state_dict:
            del state_dict['sparsity_per_head']
        if 'n_tokens' in state_dict:
            del state_dict['n_tokens']
        super().load_state_dict(state_dict, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        if self.alpha == 1.0 or self.local_attention != (-1, -1):
            attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
                self,
                qkv=qkv,
                rotary_emb=self.rotary_emb,
                local_attention=self.local_attention,
                bs=bs,
                dim=self.all_head_size,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = attn_outputs[0]
            hidden_states = self.out_drop(self.Wo(hidden_states))
        else:
            if self.use_triton_entmax:
                attn_outputs = adasplash_attention_forward(
                    self,
                    qkv=qkv,
                    rotary_emb=self.rotary_emb,
                    local_attention=self.local_attention,
                    bs=bs,
                    dim=self.all_head_size,
                    output_attentions=output_attentions,
                    alpha=self.alpha,
                    pre_niter=self.pre_iter,
                    post_niter=self.post_iter,
                    **kwargs,
                )
            else:
                attn_outputs = entmax_attention_forward(
                    self,
                    qkv=qkv,
                    rotary_emb=self.rotary_emb,
                    local_attention=self.local_attention,
                    bs=bs,
                    dim=self.all_head_size,
                    output_attentions=output_attentions,
                    alpha=self.alpha,
                    pre_niter=self.pre_iter,
                    post_niter=self.post_iter,
                    **kwargs,
                )
            hidden_states = attn_outputs[0]
            hidden_states = self.out_drop(self.Wo(hidden_states))


        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted

class CustomModernBertForMaskedLM(ModernBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for layer_id, layer in enumerate(self.model.layers):
            layer.attn = CustomModernBertAttention(config, layer_id=layer_id)


class CustomModernBertModel(ModernBertModel):
    def __init__(self, config, alpha=2.0, use_triton_entmax=False, pre_iter=5, post_iter=5, reinit_layers=True):
        config.initial_alpha = alpha
        config.alpha = alpha
        config.use_triton_entmax = use_triton_entmax
        config.pre_iter = pre_iter
        config.post_iter = post_iter
        super().__init__(config)
        if reinit_layers:
            for layer_id, layer in enumerate(self.layers):
                layer.attn = CustomModernBertAttention(config, layer_id=layer_id)


class CustomModernBertForSequenceClassification(ModernBertForSequenceClassification):
    def __init__(self, config, alpha=2.0, use_triton_entmax=False, pre_iter=5, post_iter=5, classifier_dropout=0.1):
        config.initial_alpha = alpha
        config.alpha = alpha
        config.use_triton_entmax = use_triton_entmax
        config.pre_iter = pre_iter
        config.post_iter = post_iter
        config.classifier_dropout = classifier_dropout
        super().__init__(config)
        self.model = CustomModernBertModel(config)
        self.post_init()


def get_custom_model(model_name_or_path, initial_alpha=2.0, use_triton_entmax=False,
                     pre_iter=5, post_iter=5, from_scratch=False):
    config = ModernBertConfig.from_pretrained(model_name_or_path)
    config.initial_alpha = initial_alpha
    config.use_triton_entmax = use_triton_entmax
    config.pre_iter = pre_iter
    config.post_iter = post_iter
    # load from a pretrained checkpoint or start from scratch
    if from_scratch:
        print('Training from scratch...')
        print('Config:', config)
        model = CustomModernBertForMaskedLM._from_config(config)
    else:
        print('Loading from pretrained checkpoint...')
        print('Config:', config)
        model = CustomModernBertForMaskedLM.from_pretrained(model_name_or_path, config=config)
    return model
