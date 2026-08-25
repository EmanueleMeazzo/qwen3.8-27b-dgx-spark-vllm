# vLLM 26.07 (NGC) + the DFlash2 drafter backported from vllm#52816.
#
# The upstream change is pure Python, so no kernels are rebuilt and the image
# inherits every compiled artifact from the base -- build takes seconds.
#
#   docker build -t vllm-dflash2:26.07 dflash2-backport/
#
ARG BASE=nvcr.io/nvidia/vllm:26.07-py3
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

# New modules (added wholesale by the upstream PR).
COPY files/qwen3_dflash2.py       ${SITE}/model_executor/models/qwen3_dflash2.py
COPY files/dflash2/__init__.py    ${SITE}/v1/worker/gpu/spec_decode/dflash2/__init__.py
COPY files/dflash2/speculator.py  ${SITE}/v1/worker/gpu/spec_decode/dflash2/speculator.py

# In-place edits to existing modules (registry, dispatch, gumbel, hooks).
COPY apply.py /tmp/apply.py
# apply.py asserts each anchor matched; the import proves the new module is loadable.
# Kept as its own RUN with no `|| true` so a moved anchor fails the build.
RUN python3 /tmp/apply.py ${SITE} \
 && rm /tmp/apply.py \
 && python3 -c "import vllm.model_executor.models.qwen3_dflash2"

# Belt-and-braces: the edits change mtimes so .pyc would be invalidated anyway.
RUN find ${SITE} -name '__pycache__' -path '*spec_decode*' -exec rm -rf {} + 2>/dev/null; true
