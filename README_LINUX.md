# Pollard's Rho Multi-Target (Linux / наети GPU) — бърз старт

Твоят **Pollard's Rho** GPU солвър (RCAsm `rhoStep` cubin), много таргети наведнъж,
WIF изход, multi-GPU. Търси в **ЦЕЛИЯ 256-битов keyspace — БЕЗ диапазон**.
Cubin-ът `kernel_sm89.cubin` е за **Ada (sm_89)** → RTX 4090/4080/4060Ti.

## Вход: само публични ключове
Входният файл е **списък с публични ключове, по един на ред**. Няма диапазон.
Редове, започващи с `#`, и невалидни редове се пропускат. Лимит: 134M таргета.
```
02E9F43F810784FF1E91D8BC7C4FF06BFEE935DA71D7350734C3472FE305FEF82A
03BCF7CE887FFCA5E62C9CABBDB7FFA71DC183C52C04FF4EE5EE82E0C55C39D77B
...
```
(`in.txt` служи за бърз тест с 1 ключ — редовете, дето приличат на диапазон, просто се игнорират.)

## 1. Качване веднъж в GitHub (вече направено)
Репо: https://github.com/hristoKirilov93/kangaroo-gpu

## 2. На всяка наета машина
```bash
git clone https://github.com/hristoKirilov93/kangaroo-gpu.git
cd kangaroo-gpu
bash setup_kangaroo.sh 89         # 89 = RTX 40xx (Ada) ; 120 = RTX 50xx
```
`setup_kangaroo.sh` сам намира CUDA, билдва и показва картите (`-l`).
Големият файл с ключове НЕ се качва в git — качи го отделно (scp/Jupyter).

## 3. Лов с известия
```bash
# смени NTFY_TOPIC и GPUS в run_kangaroo.sh според твоя topic и броя карти
tmux new -s rho
bash run_kangaroo.sh targets.txt      # targets.txt = списък с публични ключове
```
При намерен ключ → ntfy сигнал „влез в кутията"; ключът/WIF е във `found_rho.txt`
(не се праща по ntfy от сигурност).

Ръчно, без wrapper:
```bash
./kangaroo -gpu -gpuId 0,1,2,3,4,5,6,7 -o found_rho.txt targets.txt
# -d <битове> = DP маска (по избор; празно = auto) ; -t N = CPU нишки ; -l = списък карти
```

## Бележки
- 8 карти: `-gpuId 0,1,2,3,4,5,6,7`. 4 карти: `0,1,2,3`.
- Cubin-ът е **sm_89 (Ada)** → схемата е за **40-серия**. За 50-серия трябва sm_120 cubin.
- Билд грешка? Прати я — оправяме я (както с KeyHunt).
