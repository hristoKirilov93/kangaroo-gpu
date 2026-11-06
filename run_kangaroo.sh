#!/usr/bin/env bash
# run_kangaroo.sh — Pollard's Rho Multi-Target на всички карти + ntfy известия.
# Търси в ЦЕЛИЯ 256-битов keyspace — БЕЗ диапазон. Входът е списък с публични
# ключове (по един на ред; редове с # или невалидни се пропускат), до 134M таргета.
# При намерен ключ: сигнал в ntfy (БЕЗ ключа — той е във found файла на кутията).
# Употреба:  bash run_kangaroo.sh [файл_с_публични_ключове]   (по подразбиране targets.txt)
set -u

# ----------------------- НАСТРОЙКИ -----------------------
NTFY_TOPIC="p71-hristo-k7q2m9x4"      # <-- смени с твоя таен topic!
GPUS="0,1,2,3,4,5,6,7"                 # нагласи според броя карти (4 карти: 0,1,2,3)
INFILE="${1:-targets.txt}"             # списък с публични ключове (НЕ диапазон)
OUTFILE="found_rho.txt"                # тук падат намерените (с WIF)
DP=""                                  # -d DP битове; празно = auto
HEARTBEAT=3600                         # ntfy "жив съм" на N сек
LABEL="rho 8x4090"
# ---------------------------------------------------------

NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1)
[ -z "$CUDA_PATH" ] && command -v nvcc >/dev/null 2>&1 && CUDA_PATH=$(dirname "$(dirname "$(command -v nvcc)")")
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

ntfy(){ curl -sS -m 20 -H "Title: ${LABEL}" -H "Priority: ${2:-default}" \
  -d "$1" "$NTFY_URL" >/dev/null 2>&1; }

[ -f "$INFILE" ] || { echo "Няма файл с публични ключове: $INFILE"; exit 1; }
: > "$OUTFILE"
ntfy "🚀 ${LABEL}: старт (Rho, цял 256-бит). Таргети: ${INFILE}."
echo "Старт. Публични ключове от: $INFILE. Ctrl+C за спиране."

# watcher: щом се появи намерен ключ -> сигнал (без ключа)
( last=0
  while true; do
    n=$(grep -c . "$OUTFILE" 2>/dev/null || echo 0)
    if [ "${n:-0}" -gt "$last" ]; then
      ntfy "🎉🎉 НАМЕРЕН КЛЮЧ — ${LABEL}! Влез в кутията ВЕДНАГА — ключът/WIF е във ${OUTFILE}. (НЕ го пращам по ntfy от сигурност.)" urgent
      last=$n
    fi
    sleep 10
  done ) & WATCH=$!

# heartbeat
( while true; do sleep "$HEARTBEAT"
    ntfy "⏱ ${LABEL}: жив съм. Намерени досега: $(grep -c . "$OUTFILE" 2>/dev/null || echo 0)."
  done ) & HB=$!

trap 'kill $WATCH $HB 2>/dev/null; echo; echo "Спирам."; ntfy "⏹ ${LABEL}: спрян ръчно."; exit 130' INT TERM

DPARG=(); [ -n "$DP" ] && DPARG=(-d "$DP")
./kangaroo -gpu -gpuId "$GPUS" "${DPARG[@]}" -o "$OUTFILE" "$INFILE"
rc=$?

kill $WATCH $HB 2>/dev/null
if [ -s "$OUTFILE" ]; then
  ntfy "✅ ${LABEL}: приключи. Намерени ключове са във ${OUTFILE} — влез и ги вземи." urgent
  echo "ГОТОВО. Намерени ключове във $OUTFILE"
else
  ntfy "⚠️ ${LABEL}: процесът излезе (код $rc) без намерени. Провери кутията."
  echo "Излезе (код $rc) без намерени."
fi
