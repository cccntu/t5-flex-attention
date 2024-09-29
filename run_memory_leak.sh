## flex attention (no compile)
USE_FLEX_ATTENTION=true COMPILE_FLEX_ATTENTION=false COMPILE_WHOLE_MODEL=false MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-flex-nocompile.json PROFILE_PATH=trace-t5-flex-nocompile.json python benchmark.py
## flex attention + compile flex_attention
USE_FLEX_ATTENTION=true COMPILE_FLEX_ATTENTION=true COMPILE_WHOLE_MODEL=false MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-flex-compile-flex.json PROFILE_PATH=trace-t5-flex-compile-flex.json python benchmark.py
## flex attention + compile forward
USE_FLEX_ATTENTION=true COMPILE_FLEX_ATTENTION_MODULE=true COMPILE_WHOLE_MODEL=false MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-flex-compile-forward.json PROFILE_PATH=trace-t5-flex-compile-forward.json python benchmark.py
