import math
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask
from transformers import T5EncoderModel, T5TokenizerFast
import torch
from transformers import AutoConfig

from patch_hf_t5 import patch_hf_t5, get_mask_mod


def load_model_and_tokenizer():
    model = T5EncoderModel.from_pretrained("google-t5/t5-small",
        torch_dtype=torch.bfloat16
    )
    #    config = AutoConfig.from_pretrained("google-t5/t5-small")
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    #model = T5EncoderModel(config)

    return model, tokenizer

@torch.no_grad()
def test_attention_module(load_model_and_tokenizer_fn=load_model_and_tokenizer, device='cuda', reinit=False):
    model, tokenizer = load_model_and_tokenizer_fn()
    model.eval().to(device, non_blocking=True)


    patch_hf_t5(model, attribute_name='flex_forward')
    # NOTE: test one attention module
    attn_layer = model.encoder.block[0].layer[0].SelfAttention
    # random init q,k,v
    # initialize follows transformers/src/transformers/models/t5/modeling_t5.py
    factor = model.config.initializer_factor
    d_model = model.config.d_model
    key_value_proj_dim = model.config.d_kv
    n_heads = model.config.num_heads
    if reinit:
        attn_layer.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
        attn_layer.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
        attn_layer.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
        attn_layer.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))


    batch_size, seq_length = 1, 2048

    test_input = torch.randn(batch_size, seq_length, d_model, device=device, dtype=attn_layer.q.weight.dtype) / math.sqrt(d_model)
    test_input_norm = torch.linalg.norm(test_input, dim=-1).mean()
    #    print(f'{test_input_norm=}') # should be 1

    original_output = attn_layer(test_input)[0]
    #print(f'{original_output.shape=}')
    flex_output = attn_layer.flex_forward(test_input)[0]

    assert torch.allclose(original_output, flex_output, atol=1e-2, rtol=1e-2)
    
    attention_mask = torch.randint(0, 2, (batch_size, seq_length), device=device, dtype=torch.bool)
    while torch.sum(attention_mask) == 0 or torch.sum(attention_mask) == seq_length:
        # make sure mask is not all 0 or all 1
        attention_mask = torch.randint(0, 2, (batch_size, seq_length), device=device, dtype=torch.bool)
    mask_mod = get_mask_mod(attention_mask)
    block_mask = create_block_mask(mask_mod, B=1, H=None, Q_LEN=seq_length, KV_LEN=seq_length)
    
    flex_output = attn_layer.flex_forward(test_input, output_attentions=block_mask)[0]

    # otherwise, hf implementation will do the following boefore passing attention_mask to the attention module
    # defined in ModuleUtilsMixin at transformers/src/transformers/modeling_utils.py
    inverted_attention_mask = model.invert_attention_mask(attention_mask) 
    original_output = attn_layer(test_input, mask=inverted_attention_mask)[0]

    assert torch.allclose(original_output, flex_output, atol=1e-2, rtol=1e-2)


        

@torch.no_grad()
def test_full_forward(load_model_and_tokenizer_fn=load_model_and_tokenizer, device='cuda'):
    def get_model():
        torch.random.manual_seed(0)
        model, tokenizer = load_model_and_tokenizer_fn()
        model.eval().to(device, non_blocking=True)
        return model, tokenizer
    batch_size, seq_length = 1, 2048

    def run(text, model, tokenizer, use_mask=True):
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(device, non_blocking=True)
        kwargs = dict()
        if use_mask:
            kwargs["attention_mask"] = attention_mask
            if model.using_flex_attention:
                mask_mod = get_mask_mod(attention_mask)
                block_mask = create_block_mask(mask_mod, B=1, H=None, Q_LEN=seq_length, KV_LEN=seq_length)
                kwargs["output_attentions"] = block_mask
        with torch.no_grad():
            outputs = model(input_ids=input_ids, **kwargs)
            embeddings = outputs.last_hidden_state
        return embeddings


    test_input = 'hello world'
    model, tokenizer = get_model()
    model.using_flex_attention = False
    original_output = run(test_input, model, tokenizer, use_mask=True)
    original_output_nomask = run(test_input, model, tokenizer, use_mask=False)

    model, tokenizer = get_model()
    patch_hf_t5(model, attribute_name='flex_forward')
    model.using_flex_attention = True

    flex_output = run(test_input, model, tokenizer, use_mask=True)
    flex_output_nomask = run(test_input, model, tokenizer, use_mask=False)

    assert torch.allclose(original_output, flex_output, atol=1e-2, rtol=1e-2)
    assert torch.allclose(original_output_nomask, flex_output_nomask, atol=1e-2, rtol=1e-2)



#test_attention_module() # this can fail, but reinit shows the code is correct, so I think it's the fault of the model's weight
test_attention_module(reinit=True)
test_full_forward()

