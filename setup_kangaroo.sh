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

# намери CUDA toolkit: първо /usr/local/cuda*, иначе от самия nvcc (напр. apt/WSL -> /usr)
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1)
if [ -z "$CUDA_PATH" ] && command -v nvcc >/dev/null 2>&1; then
  CUDA_PATH=$(dirname "$(dirname "$(command -v nvcc)")")
fi
if [ -z "$CUDA_PATH" ] || [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
  echo ">>> nvcc липсва! Инсталирай CUDA toolkit (apt: nvidia-cuda-toolkit) или избери vast образ с CUDA devel. <<<"
  exit 1
fi
echo "CUDA: $CUDA_PATH"

# nvcc (CUDA<=12.x) не приема gcc>12. Ако host gcc е по-нов, ползвай g++-12 за nvcc.
CCBIN=""
GCCMAJ=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
if [ "${GCCMAJ:-0}" -gt 12 ] && command -v g++-12 >/dev/null 2>&1; then
  CCBIN="CXXCUDA=/usr/bin/g++-12"
  echo "host gcc=$GCCMAJ твърде нов за nvcc -> ползвам g++-12 за ядрото"
fi

make clean 2>/dev/null || true
make gpu=1 CCAP="$CCAP" CUDA="$CUDA_PATH" $CCBIN all

echo
echo "=== готово. Картите: ==="
LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH" ./kangaroo -l || true
echo
echo "Тест: ./kangaroo -gpu -gpuId 0 -o /tmp/t.txt in.txt   (малък диапазон от in.txt)"
echo "Лов:  bash run_kangaroo.sh <твой_таргет_файл>"
