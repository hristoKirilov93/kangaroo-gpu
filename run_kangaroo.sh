#!/usr/bin/env bash
# run_kangaroo.sh — пуска Kangaroo на всички карти + ntfy известия.
# При намерен ключ: сигнал в ntfy (БЕЗ ключа — той си стои във found файла на кутията).
# Употреба:  bash run_kangaroo.sh [таргет_файл]   (по подразбиране in.txt)
set -u

# ----------------------- НАСТРОЙКИ -----------------------
NTFY_TOPIC="p71-hristo-k7q2m9x4"          # <-- смени с твоя таен topic!
GPUS="0,1,2,3,4,5,6,7"                     # нагласи според броя карти (4 карти: 0,1,2,3)
INFILE="${1:-in.txt}"                      # диапазон + публични ключове
OUTFILE="found_kang.txt"                   # тук падат намерените (с WIF)
WORKFILE="work.kcp"                        # запазване на прогреса (resume)
DP=""                                      # -d стойност; празно = auto
SAVE_EVERY=300                             # -wi: запазвай прогреса на N сек
HEARTBEAT=3600                             # ntfy "жив съм" на N сек
LABEL="kangaroo 8x4090"
# ---------------------------------------------------------

NTFY_URL="https://ntfy.sh/${NTFY_TOPIC}"
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1)
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

ntfy(){ curl -sS -m 20 -H "Title: ${LABEL}" -H "Priority: ${2:-default}" \
  -d "$1" "$NTFY_URL" >/dev/null 2>&1; }

[ -f "$INFILE" ] || { echo "Няма таргет файл: $INFILE"; exit 1; }
: > "$OUTFILE"
ntfy "🚀 ${LABEL}: старт. Таргети: ${INFILE}."
echo "Старт. Таргети: $INFILE. Ctrl+C за спиране."

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
./kangaroo -gpu -gpuId "$GPUS" "${DPARG[@]}" -o "$OUTFILE" -w "$WORKFILE" -wi "$SAVE_EVERY" "$INFILE"
rc=$?

kill $WATCH $HB 2>/dev/null
if [ -s "$OUTFILE" ]; then
  ntfy "✅ ${LABEL}: приключи. Намерени ключове са във ${OUTFILE} — влез и ги вземи." urgent
  echo "ГОТОВО. Намерени ключове във $OUTFILE"
else
  ntfy "⚠️ ${LABEL}: kangaroo излезе (код $rc) без намерени. Провери кутията."
  echo "Излезе (код $rc) без намерени."
fi
