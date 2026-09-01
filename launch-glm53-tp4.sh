#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash-NVFP4, TP=4 across four DGX Sparks (GB10 / sm_121).
#
# Adapted from tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark with two changes that
# matter (see README):
#   1. NCCL_IB_GID_INDEX is set PER NODE. A single value is wrong on most fleets, and
#      leaving it unset makes the collective hang silently instead of failing.
#   2. The persistent_topk bind-mount is included. Without it the engine dies on any
#      request past ~24K tokens of context.
#
# Launch WORKER-FIRST, ~20 s apart: rank 3 -> 2 -> 1 -> head 0.
# Run the unconditional page-cache flusher on every node for the whole boot.
#
# Edit HOST_IP/GIDX below for your fabric. Verify each node's GID index with:
#   for i in 0 1 2 3 4 5; do #     cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i; done
NODE_RANK="${1:?usage: launch-glm53-vllm-tp4.sh <0|1|2|3>}"

IMAGE="radixark/vllm-glm53-flash:sm121-v8"
NAME="vllm_glm53"
MODEL_HOST_PATH="/var/tmp/glm-5.3-flash-nvfp4"
MODEL_PATH="/models/glm-5.3-flash-nvfp4"
CACHE_HOST_PATH="/var/tmp/glm53-vllm-cache"
HEAD_IP="10.77.1.11"
MPORT="29521"
PORT="8000"

case "$NODE_RANK" in
  # RoCEv2/IPv4 GID index differs per node - check gid_attrs/types under
  # /sys/class/infiniband/rocep1s0f0/ports/1/ before changing cabling.
  0) HOST_IP=10.77.1.11; GIDX=3; HEADLESS="" ;;      # spark-01 (head)
  1) HOST_IP=10.77.1.12; GIDX=3; HEADLESS="--headless" ;;  # spark-02
  2) HOST_IP=10.77.1.13; GIDX=3; HEADLESS="--headless" ;;  # spark-03
  3) HOST_IP=10.77.1.14; GIDX=4; HEADLESS="--headless" ;;  # spark-04
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac

# The RoCEv2 GID index is not stable across link bounces and reboots, so look it
# up instead of trusting the pin above (a stale index makes NCCL fail at init
# with "unhandled system error"). The pinned value stays as a fallback.
DETECTED_GIDX=""
for _i in 0 1 2 3 4 5 6 7; do
  _t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$_i 2>/dev/null)
  _g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$_i 2>/dev/null)
  case "$_t" in *"RoCE v2"*)
    case "$_g" in *ffff*) DETECTED_GIDX=$_i; break ;; esac ;;
  esac
done
[ -n "$DETECTED_GIDX" ] && GIDX=$DETECTED_GIDX
echo "using NCCL_IB_GID_INDEX=$GIDX"

test -f "$MODEL_HOST_PATH/config.json"
mkdir -p "$CACHE_HOST_PATH"
docker rm -f "$NAME" 2>/dev/null || true

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v $HOME/patches/sparse_attn_indexer_kpool.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=10.77.1.0/24 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 1048576 \
    --max-num-seqs 6 --block-size 2304 --moe-backend marlin --speculative-config '{"method":"mtp","num_speculative_tokens":4}' --kv-cache-dtype fp8_e4m3 --kv-cache-memory 25769803776 \
    --enforce-eager \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --chat-template /models/glm-5.3-flash-nvfp4/chat_template_mm.jinja --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP tp4"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
  echo "$NAME exited; inspect with: docker logs $NAME" >&2
  exit 1
}
