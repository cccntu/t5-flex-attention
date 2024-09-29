import time
from torch.nn.attention.flex_attention import create_block_mask
import torch.nn.attention.flex_attention
from torch.profiler import profile, ProfilerActivity
from transformers import T5EncoderModel, T5TokenizerFast
import torch
from triton.testing import do_bench
import os


# disable grad
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    device = torch.device('cpu')
    # Assuming bf16 is not available on CPU in this case, use float16 as fallback
    torch_dtype = torch.float16

print(f"Using device: {device}, dtype: {torch_dtype}")

def parse_envvar_to_bool(envvar_name, default="False"):
    return os.getenv(envvar_name, default).lower() in ("true", "1", "t")

# FLUX doesn't use attention mask, so we can disable it here
USE_ATTENTION_MASK = parse_envvar_to_bool("USE_ATTENTION_MASK", "false")
USE_FLEX_ATTENTION = parse_envvar_to_bool("USE_FLEX_ATTENTION", "false")

# whether to compile the flex attention function
COMPILE_FLEX_ATTENTION = parse_envvar_to_bool("COMPILE_FLEX_ATTENTION", "false")
COMPILE_FLEX_ATTENTION_MODULE = parse_envvar_to_bool("COMPILE_FLEX_ATTENTION_MODULE", "false")
# whether to compile the whole model or just the attention layer
COMPILE_WHOLE_MODEL = parse_envvar_to_bool("COMPILE_WHOLE_MODEL", "false")

MEMORY_SNAPSHOT_PATH = os.getenv("MEMORY_SNAPSHOT_PATH", "memory_snapshot-t5.json")
PROFILE_PATH = os.getenv("PROFILE_PATH", "trace-t5.json")


print(f"{USE_ATTENTION_MASK=}")
print(f"{USE_FLEX_ATTENTION=}")
print(f"{COMPILE_WHOLE_MODEL=}")
print(f"{COMPILE_FLEX_ATTENTION=}")


if COMPILE_FLEX_ATTENTION:
    print("Compiling flex_attention")
    torch.nn.attention.flex_attention.flex_attention = torch.compile(torch.nn.attention.flex_attention.flex_attention)
from patch_hf_t5 import patch_hf_t5, get_mask_mod

MODEL='FLUX'
def load_model_and_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder="tokenizer_2")

    model = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype
    )

    return tokenizer, model

def load_model_and_tokenizer_small():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    model = T5EncoderModel.from_pretrained("google-t5/t5-small",
        torch_dtype=torch_dtype
    )

    return tokenizer, model


class T5Embedder:
    def __init__(self, device="cuda", use_flex_attention=True, max_length=1024):
        self.use_flex_attention = use_flex_attention
        print(f"{self.use_flex_attention=}")

        tokenizer, model = load_model_and_tokenizer()
        model.eval().to(device, non_blocking=True)
        if use_flex_attention:
            if COMPILE_FLEX_ATTENTION_MODULE:
                compile_forward_fn = torch.compile
            else:
                compile_forward_fn = lambda x: x
            model = patch_hf_t5(model, compile_forward_fn=compile_forward_fn)
        if COMPILE_WHOLE_MODEL:
            # NOTE: you can try other modes here
            model = torch.compile(model, mode="max-autotune")
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length

    def __call__(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_length

        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(self.device, non_blocking=True)
        kwargs = dict()     
        if USE_ATTENTION_MASK:
            kwargs["attention_mask"] = attention_mask
            if self.use_flex_attention:
                mask_mod = get_mask_mod(attention_mask)
                block_mask = create_block_mask(mask_mod, B=1, H=None, Q_LEN=max_length, KV_LEN=max_length)
                kwargs["output_attentions"] = block_mask
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, **kwargs)
            embeddings = outputs.last_hidden_state
        return embeddings

embedder = T5Embedder(device=device, use_flex_attention=USE_FLEX_ATTENTION)

def bench_fn():
    prompt = "hello flex attention"
    embedding = embedder(prompt, max_length=1024)    
    cpu = embedding.cpu() # trigger cuda sync

# warmup compilation
for _ in range(3):
    bench_fn()

bench_time = do_bench(bench_fn)

torch.cuda.memory._record_memory_history(max_entries=100000)
num_runs = 4
total_time = 0

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for i in range(num_runs):
        start_time = time.perf_counter()
        bench_fn()
        end_time = time.perf_counter()
        total_time += end_time - start_time

average_time = total_time / num_runs
with open(f'log.txt', 'a') as f:
    f.write(f'{USE_FLEX_ATTENTION=} {USE_ATTENTION_MASK=} {COMPILE_FLEX_ATTENTION=} {COMPILE_FLEX_ATTENTION_MODULE=} {COMPILE_WHOLE_MODEL=} {MODEL=} time: {average_time} bench_time: {bench_time}\n')

torch.cuda.memory._dump_snapshot(MEMORY_SNAPSHOT_PATH)
prof.export_chrome_trace(PROFILE_PATH)
torch.cuda.memory._record_memory_history(enabled=None)
peak_memory_usage = max(evt.cpu_memory_usage for evt in prof.events())
print(f"Peak memory usage: {peak_memory_usage} bytes")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

print(f'wrote memory snapshot to {MEMORY_SNAPSHOT_PATH}')
print(f'wrote trace to {PROFILE_PATH}')