import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import RobertaForMaskedLM, RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaAttention, RobertaSelfOutput, \
    RobertaForSequenceClassification, RobertaClassificationHead


# pip install entmax
from entmax import entmax_bisect, entmax15, sparsemax

# pip install adasplash
from adasplash import triton_entmax


class CustomRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.alpha = config.initial_alpha if hasattr(config, 'initial_alpha') else 2.0
        self.use_triton_entmax = config.use_triton_entmax if hasattr(config, 'use_triton_entmax') else False
        print('>>>>>>>> ENTMAX ALPHA:', self.alpha)
        print('>>>>>>>> USE TRITON ENTMAX:', self.use_triton_entmax)
        #self.register_buffer("sparsity_per_head", torch.zeros(1, config.num_attention_heads))  # Buffer for sparsity
        self.sparsity_per_head = torch.zeros(1, config.num_attention_heads)
        # self.register_buffer("n_tokens", torch.tensor(0))  # Buffer for the number of tokens
        self.n_tokens = torch.tensor(0)

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
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Apply sparse attention
        if self.alpha == 1.0:
            attention_probs = torch.softmax(attention_scores, dim=-1)
        else:
            if self.use_triton_entmax:
                attention_probs = triton_entmax(attention_scores, alpha=self.alpha, n_iter=20, fast_math=True)
            else:
                if self.alpha == 2.0:
                    attention_probs = sparsemax(attention_scores, dim=-1)
                elif self.alpha == 1.5:
                    attention_probs = entmax15(attention_scores, dim=-1)
                else:
                    attention_probs = entmax_bisect(attention_scores, alpha=self.alpha, dim=-1, n_iter=20)
  
        # Sparsity per head calculation
        if attention_mask is not None:
            valid_mask = attention_mask == 0  # Valid positions
            valid_zero_probs = (attention_probs == 0) & valid_mask  # Sparsity pattern
            sparsity_per_head = valid_zero_probs.sum(-1).sum(-1).float() / valid_mask.sum(-1).sum(-1).float()
            self.sparsity_per_head = sparsity_per_head.detach()
            self.n_tokens = valid_mask.sum().detach()
        else:
            seq_len = attention_probs.size(-1)
            sparsity_per_head = (attention_probs == 0).sum(-1).sum(-1).float() / (seq_len * seq_len)
            self.sparsity_per_head = sparsity_per_head.detach()
            self.n_tokens = torch.tensor(seq_len * seq_len).detach()

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CustomRobertaAttention(RobertaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = CustomRobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)


class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.roberta.encoder.layer:
            layer.attention = CustomRobertaAttention(config)


class CustomRobertaModel(RobertaModel):
    def __init__(self, config, alpha=2.0, use_triton_entmax=False, pre_iter=5, post_iter=5):
        config.initial_alpha = alpha
        config.alpha = alpha
        config.use_triton_entmax = use_triton_entmax
        config.pre_iter = pre_iter
        config.post_iter = post_iter
        super().__init__(config)
        for layer in self.encoder.layer:
            layer.attention = CustomRobertaAttention(config)


class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, alpha=2.0, use_triton_entmax=False, pre_iter=5, post_iter=5):
        super().__init__(config)
        config.initial_alpha = alpha
        config.entmax_alpha = alpha
        config.use_triton_entmax = use_triton_entmax
        config.pre_iter = pre_iter
        config.post_iter = post_iter
        self.roberta = CustomRobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.post_init()


def get_custom_model(model_name_or_path, initial_alpha=2.0, use_triton_entmax=False,
                     pre_iter=5, post_iter=5, from_scratch=False):
    config = RobertaConfig.from_pretrained(model_name_or_path)
    config.initial_alpha = initial_alpha
    config.use_triton_entmax = use_triton_entmax
    config.pre_iter = pre_iter
    config.post_iter = post_iter
    # load from a pretrained checkpoint or start from scratch
    if from_scratch:
        print('Training from scratch...')
        print('Config:', config)
        model = CustomRobertaForMaskedLM._from_config(config)
    else:
        print('Loading from pretrained checkpoint...')
        print('Config:', config)
        model = CustomRobertaForMaskedLM.from_pretrained(model_name_or_path, config=config)
        # test if alpha and use_triton_entmax are set correctly
        assert model.roberta.encoder.layer[0].attention.self.alpha == initial_alpha
        assert model.roberta.encoder.layer[0].attention.self.use_triton_entmax == use_triton_entmax
    return model
