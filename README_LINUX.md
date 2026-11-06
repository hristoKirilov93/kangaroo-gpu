# Kangaroo (Linux / наети GPU) — бърз старт

JLP Kangaroo форк с RCAsm `rhoStep` cubin, много таргети наведнъж, WIF изход, multi-GPU.
Cubin-ът `kernel_sm89.cubin` е за **Ada (sm_89)** — работи на RTX 4090/4080/4060Ti.

## 1. Качване веднъж в GitHub (от твоя компютър)
```bash
cd <тази папка>
git init
git add .
git commit -m "Kangaroo linux + cubin"
git branch -M main
git remote add origin https://github.com/<твой_профил>/kangaroo-gpu.git
git push -u origin main
```
(Направи първо празно репо в github.com → New repository, БЕЗ README.)

## 2. На всяка наета машина
```bash
git clone https://github.com/<твой_профил>/kangaroo-gpu.git
cd kangaroo-gpu
bash setup_kangaroo.sh 89         # 89 = RTX 40xx ; 120 = RTX 50xx
```
`setup_kangaroo.sh` билдва и показва картите (`-l`).

## 3. Таргет файл
Голям `targets.txt` НЕ се качва в git (твърде голям). Качи го отделно на кутията
(scp/Jupyter). Формат:
```
<START_hex>
<END_hex>
<pubkey1>
<pubkey2>
...
```
(в `in.txt` има малък пример.)

## 4. Лов с известия
```bash
# смени NTFY_TOPIC и GPUS в run_kangaroo.sh според твоя topic и броя карти
tmux new -s kang
bash run_kangaroo.sh targets.txt
```
При намерен ключ → ntfy сигнал „влез в кутията"; ключът/WIF е във `found_kang.txt`
(не се праща по ntfy от сигурност). `-w work.kcp` пази прогреса за resume.

## Бележки
- 8 карти: `-gpuId 0,1,2,3,4,5,6,7`. 4 карти: `0,1,2,3`.
- Blackwell (5090): cubin-ът е sm_89; за 50-серия трябва sm_120 cubin (нямаме такъв готов) —
  затова засега тази схема е за **40-серия**.
- Ако билдът даде грешка — прати я, оправяме я (като с KeyHunt).
