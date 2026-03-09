FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6

# Override the installed tensorrt_llm Python package with our custom fork,
# while preserving compiled C++/CUDA bindings (.so) and libs/ from the base image.

ARG SITE=/usr/local/lib/python3.12/dist-packages/tensorrt_llm
ARG REPO=https://github.com/phatvo9/TensorRT-LLM.git
ARG BRANCH=fix-structured-output

# Clone our fork and overlay the serve/ directory onto the installed package.
# Only serve/ contains our fixes (openai_protocol.py, harmony_adapter.py,
# postprocess_handlers.py). Other directories are left as rc6 to avoid
# binding incompatibilities between rc7 Python code and rc6 compiled bindings.
RUN git clone --depth 1 --branch ${BRANCH} --filter=blob:none --sparse ${REPO} /tmp/trtllm-src && \
    cd /tmp/trtllm-src && \
    git sparse-checkout set tensorrt_llm/serve && \
    cp -r /tmp/trtllm-src/tensorrt_llm/serve/* ${SITE}/serve/ && \
    rm -rf /tmp/trtllm-src

# Embed config.yaml so the image is self-contained
RUN cat > /app/config.yaml <<'EOF'
max_batch_size: 10
max_num_tokens: 131072
max_seq_len: 131072
tensor_parallel_size: 1
moe_expert_parallel_size: 1
trust_remote_code: true
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
  speculative_model: nvidia/gpt-oss-120b-Eagle3
cuda_graph_config:
  max_batch_size: 10
sampler_type: TorchSampler
moe_config:
  backend: TRTLLM
guided_decoding_backend: xgrammar
reasoning_parser: gpt_oss
EOF

# Default environment variables
ENV TRTLLM_ENABLE_PDL=1

# Default command — override model/port as needed
CMD ["trtllm-serve", "openai/gpt-oss-120b", \
     "--host", "0.0.0.0", \
     "--port", "23339", \
     "--trust_remote_code", \
     "--config", "/app/config.yaml"]
