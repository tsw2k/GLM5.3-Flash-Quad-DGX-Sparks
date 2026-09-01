#!/usr/bin/env bash
set -euo pipefail
cd ~/glm53/docker
R=radixark/vllm-glm53-flash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130
echo "=== v1 -> sm121-nope-mla ==="; docker build -f Dockerfile.glm53-sm121    -t $R:sm121-nope-mla   .
echo "=== v3 -> sm121-fi618 ===";    docker build -f Dockerfile.glm53-sm121-v3 -t $R:sm121-fi618      .
echo "=== v4 -> sm121-fi618-nccl ==="; docker build -f Dockerfile.glm53-sm121-v4 -t $R:sm121-fi618-nccl .
echo "=== v5 -> sm121-final ===";    docker build -f Dockerfile.glm53-sm121-v5 -t $R:sm121-final      .
echo "=== v6 -> sm121-v6 ===";       docker build -f Dockerfile.glm53-sm121-v6 -t $R:sm121-v6         .
echo "=== v7 -> sm121-v7 ===";       docker build -f Dockerfile.glm53-sm121-v7 -t $R:sm121-v7         .
echo "=== v8 -> sm121-v8 ===";       docker build -f Dockerfile.glm53-sm121-v8 -t $R:sm121-v8         .
echo "BUILD-COMPLETE"; docker images | grep sm121
