CUDA_VISIBLE_DEVICES=0 vllm serve /home/share/models/Qwen2.5-14B-Instruct \
           --served-model-name  Qwen2.5-14B-Instruct \
           --port 8000 \
           --host  0.0.0.0 \
           --gpu-memory-utilization 0.8 \
           --trust-remote-code \
           --max-model-len 20000 \
           --enable-prefix-caching \