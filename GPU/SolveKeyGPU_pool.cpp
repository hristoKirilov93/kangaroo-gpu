// ============================================================================
// ОБЕДИНЕНИ DP ЗА МНОГО GPU  (drop-in за Kangaroo::SolveKeyGPU в Rho версията)
// ----------------------------------------------------------------------------
// Всяка карта работи в своя нишка (Run() вече ги пуска). Вместо всяка карта да
// лови колизии в своята GPU таблица, тук всички карти връщат своите DP на хоста
// и ги вкарват в ЕДНА обща хеш-таблица под ghMutex. Така колизия се лови и между
// точки от РАЗЛИЧНИ карти -> истинско N-кратно скалиране към решаването.
//
// Какво да добавиш в Kangaroo.cpp:
//   1. #include <unordered_map>  (горе)
//   2. Блока с gPoolDP по-долу (файлов обхват, над SolveKeyGPU)
//   3. Смени тялото на Kangaroo::SolveKeyGPU с това по-долу
// GPUEngine вече има SetPoolMode(true) и LaunchPool(...).
// ============================================================================

#include <unordered_map>

// ---- обща таблица с DP между всички карти (пази се с ghMutex) ----
namespace {
struct DpKey { uint64_t x0, x1;
    bool operator==(const DpKey& o) const { return x0 == o.x0 && x1 == o.x1; } };
struct DpKeyHash { size_t operator()(const DpKey& k) const {
    uint64_t h = k.x0 ^ (k.x1 * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32; return (size_t)h; } };
struct DpVal { uint64_t a[4]; uint32_t st; };
static std::unordered_map<DpKey, DpVal, DpKeyHash> gPoolDP;   // ОБЩА за всички GPU нишки
}

// ----------------------------------------------------------------------------

void Kangaroo::SolveKeyGPU(TH_PARAM* ph) {

#ifdef WITHGPU
    std::vector<GPUEngine::GpuDP> dps;

    GPUEngine* gpu = new GPUEngine(ph->gridSizeX, ph->gridSizeY, ph->gpuId, 65536);
    gpu->SetPoolMode(true);                      // <-- обединени DP на хоста

    ::printf("GPU: %s (%d MB used)\n", gpu->deviceName.c_str(), gpu->GetMemory());
    double t0 = Timer::get_tick();

    if (ph->px == NULL) {
        uint64_t nbThread = gpu->GetNbThread();
        ph->px = new Int[ph->nbKangaroo];
        ph->py = new Int[ph->nbKangaroo];
        ph->a  = new Int[ph->nbKangaroo];
        ph->state = new uint32_t[ph->nbKangaroo];
        for (uint64_t i = 0; i < nbThread; i++) {
            CreateHerd(GPU_GRP_SIZE,
                &(ph->px[i * GPU_GRP_SIZE]), &(ph->py[i * GPU_GRP_SIZE]),
                &(ph->a[i * GPU_GRP_SIZE]), &(ph->state[i * GPU_GRP_SIZE]),
                TAME, false);
        }
    }

    gpu->SetParams(dMask, jCGtable, jumpPointx, jumpPointy);
    gpu->SetKangaroos(ph->px, ph->py, ph->a, ph->state);
    safe_delete_array(ph->px);
    safe_delete_array(ph->py);
    safe_delete_array(ph->a);
    if (ph->state) { delete[] ph->state; ph->state = NULL; }

    gpu->callKernel();
    double t1 = Timer::get_tick();
    ::printf("SolveKeyGPU Thread GPU#%d: 2^%.2f walks [%.1fs]\n",
        ph->gpuId, log2((double)ph->nbKangaroo), (t1 - t0));
    ph->hasStarted = true;

    while (!endOfSearch) {

        if (!gpu->LaunchPool(dps)) break;        // пуска ядрото и връща DP от предния пуск
        counters[0x80L + ph->threadId] += ph->nbKangaroo * NB_RUN;

        if (!dps.empty()) {
            LOCK(ghMutex);
            for (size_t i = 0; i < dps.size() && !endOfSearch; i++) {
                const GPUEngine::GpuDP& d = dps[i];
                DpKey k{ d.x[0], d.x[1] };
                auto it = gPoolDP.find(k);
                if (it == gPoolDP.end()) {
                    DpVal v; v.a[0]=d.a[0]; v.a[1]=d.a[1]; v.a[2]=d.a[2]; v.a[3]=d.a[3]; v.st=d.st;
                    gPoolDP.emplace(k, v);
                } else {
                    const DpVal& o = it->second;
                    bool same = (o.a[0]==d.a[0] && o.a[1]==d.a[1] &&
                                 o.a[2]==d.a[2] && o.a[3]==d.a[3] && o.st==d.st);
                    if (!same) {
                        // РЕАЛНА колизия (същата точка, различни скалари) -> решаваме
                        COLLISION c;
                        c.x[0] = d.x[0]; c.x[1] = d.x[1];
                        c.a1[0]=d.a[0]; c.a1[1]=d.a[1]; c.a1[2]=d.a[2]; c.a1[3]=d.a[3]; c.st1=d.st;
                        c.a2[0]=o.a[0]; c.a2[1]=o.a[1]; c.a2[2]=o.a[2]; c.a2[3]=o.a[3]; c.st2=o.st;
                        if (!SolveCollision(c))
                            collisionInSameHerd++;
                    }
                    // ако same -> дубликат, пропускаме
                }
            }
            UNLOCK(ghMutex);
        }

        // --- рестарт на заклещените разходки (per-карта, както преди) ---
        {
            static thread_local std::vector<uint32_t> resetIdx;
            uint32_t nReset = gpu->GetResetList(resetIdx);
            if (nReset > 0) {
                for (size_t r = 0; r < resetIdx.size(); r++) {
                    Int n_px, n_py, n_a; uint32_t n_st;
                    CreateHerd(1, &n_px, &n_py, &n_a, &n_st, (int)(resetIdx[r] & 1), false);
                    gpu->SetKangaroo(resetIdx[r], &n_px, &n_py, &n_a, n_st);
                }
            }
        }
    }

    delete gpu;
#else
    ph->hasStarted = true;
#endif
    ph->isRunning = false;
}