#!/usr/bin/env bash
# setup_kangaroo.sh — билд на наета Linux машина.
# Употреба:  bash setup_kangaroo.sh [CCAP]
#   CCAP = 89  за RTX 40xx (Ada, 4090/4080) — ПО ПОДРАЗБИРАНЕ
#          120 за RTX 50xx (Blackwell, 5090/5080)
set -e
CCAP="${1:-89}"
echo "=== Kangaroo build, CCAP=$CCAP ==="

apt-get update || true
apt-get install -y build-essential git

# намери CUDA toolkit (vast образите обикновено го имат)
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1)
if [ -z "$CUDA_PATH" ] || [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
  echo ">>> nvcc липсва! Избери vast образ с CUDA toolkit (devel), после пусни пак. <<<"
  exit 1
fi
echo "CUDA: $CUDA_PATH"

make clean 2>/dev/null || true
make gpu=1 CCAP="$CCAP" CUDA="$CUDA_PATH" all

echo
echo "=== готово. Картите: ==="
LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH" ./kangaroo -l || true
echo
echo "Тест: ./kangaroo -gpu -gpuId 0 -o /tmp/t.txt in.txt   (малък диапазон от in.txt)"
echo "Лов:  bash run_kangaroo.sh <твой_таргет_файл>"
