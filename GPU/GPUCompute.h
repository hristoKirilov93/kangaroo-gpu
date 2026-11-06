#ifndef GPUCOMPUTEH
#define GPUCOMPUTEH

#include <stdint.h>
#include <cuda_runtime.h>
#include "../Constants.h"
#include "GPUMath.h"
#include "GPUGLV.h"
#include "GPUHashTable.h"
#include "GPUEngine.h"   // STAT_SLOTS, RESET_MAX

// ---------------------------------------------------------------------------
// Мулти-таргет Ро на Полард. DP таблица в GPU паметта. GLV + negation map.
//
// Състояние на разходка:  R = phi^e( +-( A*G + B*T_tIdx ) )
//   a_true = s * lambda^e * A  (mod n)
//   b_true = s * lambda^e * B  (mod n)
//
// Скаларът се обновява чрез предварително мащабирана таблица
//   jCG[e][i] = lambda^(-e) * c_i  (mod n)
// -> нула модулни умножения на стъпка.
//
// ---------------------------------------------------------------------------
// ИЗБОР НА БИТОВЕ ЗА ИНДЕКСА НА СКОКА
//
// jmp се взима от битове 32..39 на x. НЕ от долните битове -- те участват
// в DP маската и всяка distinguished point би имала jmp = 0, което
// деформира разходката.
//
// ---------------------------------------------------------------------------
// БЕЗПЛОДНИ ЦИКЛИ
//
// При negation map разходката може да попадне в 2-цикъл A<->B:
//   C канонична (y_C <= p/2), скок i дава R = C + S_i. Ако y_R > p/2,
//   канонизацията негира -> R' = -R, но x не се променя, затова следващият
//   индекс пак идва от x_R. Ако излезе същото i:
//        -R + S_i = -(C + S_i) + S_i = -C
//   а y_{-C} >= p/2 -> пак негация -> обратно в C.
//
// 1/(2r) е честотата на ВЛИЗАНЕ на стъпка. Без escape капанът е абсорбиращ.
//
// Escape се прилага САМО от по-малката точка на цикъла, за да остане
// разходката детерминистична функция САМО на точката -- иначе две разходки,
// стъпили на едно място с различна история, се разделят и колизията се губи
// (Bernstein-Lange-Schwabe, "On the correct use of the negation map").
//
// Escape чупи всяка отделна двойка, но някои разходки влизат в ДЪЛГА
// периодична орбита и escape ги мести от двойка в двойка в рамките на същия
// затворен кръг. Такава разходка не изследва нищо. Понеже в Ро позицията
// няма натрупана стойност, тя се РЕСТАРТИРА: ядрото я маркира, хостът я
// подменя през SetKangaroo.
// ---------------------------------------------------------------------------

// Задействания на детектора в едно извикване на ядрото, след които
// разходката се смята за безнадеждно заклещена. Нормално е 1-3.
#define CYC_RESET_THRESHOLD 32

__device__ void ComputeKangaroos(uint64_t* kangaroos,
    GHT_ENTRY* ghtTable,
    uint32_t maxFound,
    uint32_t* out,
    uint64_t dpMask,
    unsigned long long* stat,
    uint32_t* resetBuf,
    uint32_t  resetMax) {

    uint64_t px[GPU_GRP_SIZE][4];
    uint64_t py[GPU_GRP_SIZE][4];
    uint64_t a_scal[GPU_GRP_SIZE][4];
    uint32_t st[GPU_GRP_SIZE];

    uint64_t cyc0[GPU_GRP_SIZE];   // x0 отпреди 1 стъпка
    uint64_t cyc1[GPU_GRP_SIZE];   // x0 отпреди 2 стъпки
    uint32_t cycRun[GPU_GRP_SIZE]; // последователни стъпки в цикъл
    uint32_t cycTot[GPU_GRP_SIZE]; // общо задействания за тази разходка

    uint64_t dx[GPU_GRP_SIZE][4];
    uint64_t dy[4], rx[4], ry[4], _s[4], _p[4];
    uint32_t jmp;

    uint32_t nCycle = 0, nDP = 0, nFull = 0, nEscape = 0, nReset = 0;

    LoadKangaroosRho(kangaroos, px, py, a_scal, st);

#pragma unroll
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        cyc0[g] = 0ULL; cyc1[g] = 0ULL;
        cycRun[g] = 0u; cycTot[g] = 0u;
    }

    for (int run = 0; run < NB_RUN; run++) {

        __syncthreads();

        // ---- 1. dx = px - jPx ----
#pragma unroll
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            jmp = (uint32_t)(px[g][0] >> 32) & (NB_JUMP - 1);
            ModSub256(dx[g], px[g], jPx[jmp]);

            // Нулево dx би повредило ЦЯЛАТА група при груповата инверсия
            // (продуктовата верига се занулява). Заместваме с 1 -- само
            // тази разходка получава грешен резултат.
            if ((dx[g][0] | dx[g][1] | dx[g][2] | dx[g][3]) == 0ULL) {
                dx[g][0] = 1ULL; dx[g][1] = 0ULL;
                dx[g][2] = 0ULL; dx[g][3] = 0ULL;
            }
        }

        // ---- 2. Групова модулна инверсия (един ModInv за цялата група) ----
        _ModInvGrouped(dx);

        __syncthreads();

#pragma unroll
        for (int g = 0; g < GPU_GRP_SIZE; g++) {

            jmp = (uint32_t)(px[g][0] >> 32) & (NB_JUMP - 1);

            // ---- 3. Събиране на точки ----
            ModSub256(dy, py[g], jPy[jmp]);
            _ModMult(_s, dy, dx[g]);
            _ModSqr(_p, _s);
            ModSub256(rx, _p, jPx[jmp]);
            ModSub256(rx, px[g]);
            ModSub256(ry, px[g], rx);
            _ModMult(ry, _s);
            ModSub256(ry, py[g]);
            Load256(px[g], rx);
            Load256(py[g], ry);

            // ---- 4. Скаларът, ПРЕДИ канонизацията ----
            uint32_t curE = (st[g] >> 29) & 3u;
            uint32_t curNeg = (st[g] >> 28) & 1u;
            if (curNeg == 0) AddModOrder256(a_scal[g], jCG[curE][jmp]);
            else             SubModOrder256(a_scal[g], jCG[curE][jmp]);

            // ---- 5. Канонизация ----
            uint32_t dE, dNeg;
            _CanonicalClass(px[g], py[g], &dE, &dNeg);
            uint32_t newE = curE + dE; if (newE >= 3u) newE -= 3u;
            st[g] = (st[g] & 0x0FFFFFFFu) | (newE << 29) | ((curNeg ^ dNeg) << 28);

            // ---- 6. Безплоден 2-цикъл ----
            uint64_t tag = px[g][0];
            uint64_t partner = cyc0[g];
            bool twoCycle = (cyc1[g] != 0ULL) && (tag == cyc1[g]);

            if (twoCycle) {
                nCycle++;
                cycTot[g]++;
                cycRun[g]++;

                if (cycTot[g] == CYC_RESET_THRESHOLD) {
                    uint32_t slot = atomicAdd(resetBuf, 1u);
                    if (slot < resetMax) {
                        uint64_t kIdx = (uint64_t)blockIdx.x *
                            ((uint64_t)blockDim.x * GPU_GRP_SIZE)
                            + (uint64_t)g * blockDim.x
                            + (uint64_t)IDX;
                        resetBuf[1 + slot] = (uint32_t)kIdx;
                    }
                    nReset++;
                }
            }
            else {
                cycRun[g] = 0u;
            }

            cyc1[g] = cyc0[g];
            cyc0[g] = tag;

            // Escape само от по-малката точка на цикъла.
            if (twoCycle && tag < partner) {

                nEscape++;
                uint32_t esc = (uint32_t)((px[g][1] >> 17) ^ (partner >> 23))
                    & (NB_JUMP - 1u);

                uint64_t edx[4], edy[4], es[4], ep[4], erx[4], ery[4];
                ModSub256(edx, px[g], jPx[esc]);

                if ((edx[0] | edx[1] | edx[2] | edx[3]) != 0ULL) {

                    uint64_t inv[5];
                    Load256(inv, edx); inv[4] = 0;
                    _ModInv(inv);

                    ModSub256(edy, py[g], jPy[esc]);
                    _ModMult(es, edy, inv);
                    _ModSqr(ep, es);
                    ModSub256(erx, ep, jPx[esc]);
                    ModSub256(erx, px[g]);
                    ModSub256(ery, px[g], erx);
                    _ModMult(ery, es);
                    ModSub256(ery, py[g]);

                    Load256(px[g], erx);
                    Load256(py[g], ery);

                    uint32_t eE = (st[g] >> 29) & 3u;
                    uint32_t eNeg = (st[g] >> 28) & 1u;
                    if (eNeg == 0) AddModOrder256(a_scal[g], jCG[eE][esc]);
                    else           SubModOrder256(a_scal[g], jCG[eE][esc]);

                    uint32_t cE, cN;
                    _CanonicalClass(px[g], py[g], &cE, &cN);
                    uint32_t nE2 = eE + cE; if (nE2 >= 3u) nE2 -= 3u;
                    st[g] = (st[g] & 0x0FFFFFFFu) | (nE2 << 29) | ((eNeg ^ cN) << 28);

                    cyc0[g] = px[g][0];
                    cyc1[g] = 0ULL;
                    cycRun[g] = 0u;
                }
            }

            // ---- 7. Distinguished Point ----
            if ((px[g][0] & dpMask) == 0) {

                nDP++;
                GHT_ENTRY found;
                int r = GhtInsert(ghtTable, px[g], a_scal[g], st[g], &found);

                if (r == GHT_FULL) {
                    nFull++;
                }
                else if (r == GHT_COLLISION) {
                    uint32_t pos = atomicAdd(out, 1);
                    if (pos < maxFound) {
                        uint32_t base = 1 + pos * COLL_ITEM_SIZE32;
                        out[base + 0] = (uint32_t)(px[g][0]);
                        out[base + 1] = (uint32_t)(px[g][0] >> 32);
                        out[base + 2] = (uint32_t)(px[g][1]);
                        out[base + 3] = (uint32_t)(px[g][1] >> 32);
                        for (int k = 0; k < 8; k++)
                            out[base + 4 + k] = ((uint32_t*)a_scal[g])[k];
                        out[base + 12] = st[g];
                        out[base + 13] = (uint32_t)(found.a0);
                        out[base + 14] = (uint32_t)(found.a0 >> 32);
                        out[base + 15] = (uint32_t)(found.a1);
                        out[base + 16] = (uint32_t)(found.a1 >> 32);
                        out[base + 17] = (uint32_t)(found.a2);
                        out[base + 18] = (uint32_t)(found.a2 >> 32);
                        out[base + 19] = (uint32_t)(found.a3);
                        out[base + 20] = (uint32_t)(found.a3 >> 32);
                        out[base + 21] = found.meta;
                    }
                }
            }
        }
    }

    __syncthreads();

    StoreKangaroosRho(kangaroos, px, py, a_scal, st);

    // Минимални броячи -- по един atomic на нишка за цялото ядро.
    // stat[0] cycle | stat[1] DP | stat[2] full | stat[10] escape | stat[14] reset
    if (nCycle)  atomicAdd(stat + 0, (unsigned long long)nCycle);
    if (nDP)     atomicAdd(stat + 1, (unsigned long long)nDP);
    if (nFull)   atomicAdd(stat + 2, (unsigned long long)nFull);
    if (nEscape) atomicAdd(stat + 10, (unsigned long long)nEscape);
    if (nReset)  atomicAdd(stat + 14, (unsigned long long)nReset);
}

#endif // GPUCOMPUTEH