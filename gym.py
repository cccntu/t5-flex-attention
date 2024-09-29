"""Implementation of an T5 relative position bias score mod.
This file is copied and modified from https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/mods/alibi.py
Which is licensed under the BSD 3-Clause License.

To run his you need to follow the install instructions on https://github.com/pytorch-labs/attention-gym
"""

import os
from functools import partial
from torch.nn.attention.flex_attention import _score_mod_signature
from transformers import T5EncoderModel, T5TokenizerFast
import torch


def generate_t5_bias(embedding_weight: torch.Tensor, bucket_fn) -> _score_mod_signature:
    """Returns an t5 bias score_mod given the number of heads H

    Args:
        embedding_weight: embedding weight of the t5 model
        # model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight

    Returns:
        bucket bias score_mod
    """

    def t5_score_mod_bucket(score, b, h, q_idx, kv_idx):
        """for visualizing the bucket ids instead of the bias"""

        relative_position = kv_idx - q_idx
        bucket_id = bucket_fn(relative_position=relative_position)
        return score + bucket_id

    # each layer share the same embedding
    def t5_score_mod(score, b, h, q_idx, kv_idx):
        relative_position = q_idx - kv_idx
        bucket_id = bucket_fn(relative_position=relative_position)
        bias = embedding_weight[bucket_id, h]
        return score + bias

    return t5_score_mod_bucket, t5_score_mod


def main(device: str = "cpu"):
    """Visualize the attention scores alibi bias score mod.

    Args:
        device (str): Device to use for computation. Defaults
    """
    import torch
    from attn_gym import visualize_attention_scores

    model = T5EncoderModel.from_pretrained("google-t5/t5-small").to(device)

    bucket_fn = partial(
        model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket,
        bidirectional=(not model.config.is_decoder),
        num_buckets=model.config.relative_attention_num_buckets,
        max_distance=model.config.relative_attention_max_distance,
    )

    weight = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    n_head = model.config.num_heads
    B, H, SEQ_LEN, HEAD_DIM = 1, n_head, 128, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    
    t5_score_mod_bucket, t5_score_mod = generate_t5_bias(
        weight.to(device),
        bucket_fn=bucket_fn,
    )
    os.makedirs("output_images", exist_ok=True)
    for i in range(weight.shape[1]):
        visualize_attention_scores(
            query,
            key,
            score_mod=t5_score_mod,
            device=device,
            name=f"output_images/t5_score_mod_bucket_head_{i}",
            head_idx=i,
        )

if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)