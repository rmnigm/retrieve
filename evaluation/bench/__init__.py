import os

# Read before any bench module imports torch._inductor, whose import writes its default here.
GIVEN_INDUCTOR_CACHE = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
