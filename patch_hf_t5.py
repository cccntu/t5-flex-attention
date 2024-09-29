from functools import partial
from torch.nn.attention.flex_attention import flex_attention, _score_mod_signature
from transformers import T5EncoderModel
import torch

def generate_t5_bias(embedding_weight: torch.Tensor, bucket_fn) -> _score_mod_signature:
    """Returns an t5 bias score_mod given the number of heads H

    Args:
        embedding_weight: embedding weight of the t5 model
        # model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight

    Returns:
        bucket bias score_mod
    """
    # NOTE: each layer share the same embedding, this score_mod can be shared by all layers
    def t5_score_mod(score, b, h, q_idx, kv_idx):
        relative_position = kv_idx - q_idx  # (note: not the other way around, this is the same as hf implementation)
        bucket_id = bucket_fn(relative_position=relative_position)
        bias = embedding_weight[bucket_id, h]
        return score + bias

    return t5_score_mod

def get_mask_mod(attention_mask):
    attention_mask = attention_mask.bool()
    # 1 is to be attended, 0 is masked
    def mask_mod(b, h, q_idx, kv_idx):
        return attention_mask[b, kv_idx]
    return mask_mod

@torch.no_grad()
def forward(
    # self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False, # NOTE: here we use output_attentions to pass in the block_mask
    self=None,  # I'm monkey patching the method here, so I need to pass the self object
    score_mod=None,
):
    """
    Simplified self-attention mechanism using only q, k, v, o projections and dot-product attention.
    """
    batch_size, seq_length, _ = hidden_states.shape

    # Compute query, key, and value states
    query_states = (
        self.q(hidden_states).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    )
    key_states = (
        self.k(hidden_states).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    )
    value_states = (
        self.v(hidden_states).view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    )
    # q: batch_size, n_heads, seq_length, key_value_proj_dim
    # k: batch_size, n_heads, seq_length, key_value_proj_dim
    # v: batch_size, n_heads, seq_length, key_value_proj_dim

    # NOTE: here I'm using the output_attentions argument to pass the block_mask

    if not isinstance(output_attentions, bool):
        block_mask = output_attentions
    else:
        block_mask = None

    attn_output = flex_attention(
        query_states,
        key_states,
        value_states,
        scale=1, # NOTE: T5 doesn't use scale in attention, need to pass 1 here
        score_mod=score_mod,
        block_mask=block_mask,
    ) 
    attn_output = (
        attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_dim)
    )
    attn_output = self.o(attn_output)

    present_key_value_state = None
    position_bias = None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
    attention_probs = None

    if output_attentions:
        outputs += (attention_probs,)
    return outputs


def patch_hf_t5(model: T5EncoderModel, attribute_name='forward', compile_forward_fn=lambda x: x):
    """ Patch a T5EncoderModel to use flex attention
    currently doesn't support T5 decoder
    """
    first_attn_layer = model.encoder.block[0].layer[0].SelfAttention
    pos_emb_weight = first_attn_layer.relative_attention_bias.weight.data
    #n_bucket, n_head = pos_emb_weight.shape
    bucket_fn = partial(
        first_attn_layer._relative_position_bucket,  # NOTE: this is the method that calculates the bucket ids
        bidirectional=(not model.config.is_decoder),
        num_buckets=model.config.relative_attention_num_buckets,
        max_distance=model.config.relative_attention_max_distance,
    )
    pos_emb_weight.requires_grad = False  # NOTE: Captured buffers that require grad are not yet supported by FlexAttention (as of Sep 2024, torch nighlty)
    t5_relpe_score_mod = generate_t5_bias(pos_emb_weight.detach().clone(), bucket_fn=bucket_fn)

    for block in model.encoder.block:
        # NOTE: monkey patch the forward method of all attention layers
        setattr(
            block.layer[0].SelfAttention,
            attribute_name,
            partial(
                compile_forward_fn(forward),
                self=block.layer[0].SelfAttention,
                score_mod=t5_relpe_score_mod,
            ),
        )
    return model