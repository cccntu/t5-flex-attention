# baseline
USE_FLEX_ATTENTION=false COMPILE_WHOLE_MODEL=false MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-noflex-nocompile.json PROFILE_PATH=trace-t5-noflex-nocompile.json python benchmark.py
# baseline + compile
USE_FLEX_ATTENTION=false COMPILE_WHOLE_MODEL=true MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-noflex-compile.json PROFILE_PATH=trace-t5-noflex-compile.json python benchmark.py
# flex attention + compile whole model
USE_FLEX_ATTENTION=true COMPILE_WHOLE_MODEL=true MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-flex-compile.json PROFILE_PATH=trace-t5-flex-compile.json python benchmark.py

export USE_ATTENTION_MASK=true

# baseline
USE_FLEX_ATTENTION=false COMPILE_WHOLE_MODEL=false MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-noflex-nocompile.json PROFILE_PATH=trace-t5-noflex-nocompile.json python benchmark.py
# baseline + compile
USE_FLEX_ATTENTION=false COMPILE_WHOLE_MODEL=true MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-noflex-compile.json PROFILE_PATH=trace-t5-noflex-compile.json python benchmark.py
# flex attention + compile whole model
USE_FLEX_ATTENTION=true COMPILE_WHOLE_MODEL=true MEMORY_SNAPSHOT_PATH=memory_snapshot-t5-flex-compile.json PROFILE_PATH=trace-t5-flex-compile.json python benchmark.py

