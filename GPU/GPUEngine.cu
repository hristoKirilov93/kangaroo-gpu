/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* (Modified for Pollard's Rho, multi-target, GLV, GPU-resident DP table)
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "../Timer.h"
#include "GPUMath.h"
#include "GPUGLV.h"
#include "GPUHashTable.h"
#include "GPUCompute.h"

using namespace std;

// ---------------------------------------------------------------------------
// SASS ядро rhoStep (kernel_sm89.cubin). Параметрите са 0x70 байта и ТРЯБВА
// да съвпадат с шаблона в kernel.cu и с отместванията в rho_step.asm (0x160+).
// ---------------------------------------------------------------------------
#pragma pack(push, 8)
struct RhoSassParams {
    uint64_t* kangaroos; void* ght; uint32_t* out; unsigned long long* stat;
    uint32_t* resetBuf; uint64_t* jumpXY; uint64_t* jumpCG; uint64_t* scratch;
    uint64_t dpMask; uint32_t maxFound, resetMax, nbRun, grpSize;
    uint64_t* dpBuf; uint32_t dpMax, pad0; uint64_t* cycBuf;
};
#pragma pack(pop)
static_assert(sizeof(RhoSassParams) == 0x70, "RhoSassParams size");
static_assert(offsetof(RhoSassParams, dpMask) == 0x40, "dpMask offset");
static_assert(offsetof(RhoSassParams, dpBuf) == 0x58, "dpBuf offset");
static_assert(offsetof(RhoSassParams, cycBuf) == 0x68, "cycBuf offset");

// ---------------------------------------------------------------------------
// DP от rhoStep -> GHT. Всеки запис минава през GhtInsert; колизиите се пишат
// в out ТОЧНО както в ComputeKangaroos, стъпка 7 (22 думи).
// dpBuf: u32 брояч на [0]; запис i на байт 64 + 64*i: x0 x1 a0 a1 a2 a3 st pad
// ---------------------------------------------------------------------------
__global__ void ghtInsertDP(const uint64_t* dpBuf, GHT_ENTRY* table, uint32_t* out,
    unsigned long long* stat, uint32_t dpMax, uint32_t maxFound) {
    uint32_t cnt = *(const uint32_t*)dpBuf;
    uint32_t n = cnt < dpMax ? cnt : dpMax;
    uint32_t nFull = 0;
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint64_t* r = dpBuf + 8 + (uint64_t)i * 8;
        uint64_t x[2] = { r[0], r[1] };
        uint64_t a[4] = { r[2], r[3], r[4], r[5] };
        uint32_t st = (uint32_t)r[6];
        GHT_ENTRY found;
        int res = GhtInsert(table, x, a, st, &found);
        if (res == GHT_FULL) {
            nFull++;
        }
        else if (res == GHT_COLLISION) {
            uint32_t pos = atomicAdd(out, 1u);
            if (pos < maxFound) {
                uint32_t base = 1 + pos * COLL_ITEM_SIZE32;
                out[base + 0] = (uint32_t)(x[0]);
                out[base + 1] = (uint32_t)(x[0] >> 32);
                out[base + 2] = (uint32_t)(x[1]);
                out[base + 3] = (uint32_t)(x[1] >> 32);
                for (int k = 0; k < 8; k++)
                    out[base + 4 + k] = ((uint32_t*)a)[k];
                out[base + 12] = st;
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
    if (nFull) atomicAdd(stat + 2, (unsigned long long)nFull);
    if (blockIdx.x == 0 && threadIdx.x == 0 && cnt) atomicAdd(stat + 1, (unsigned long long)cnt);
}

// ---------------------------------------------------------------------------

__global__ void comp_kangaroos(uint64_t* kangaroos, GHT_ENTRY* ght,
    uint32_t maxFound, uint32_t* found,
    uint64_t dpMask, unsigned long long* stat,
    uint32_t* resetBuf, uint32_t resetMax) {
    int xPtr = (blockIdx.x * blockDim.x * GPU_GRP_SIZE) * KSIZE;
    ComputeKangaroos(kangaroos + xPtr, ght, maxFound, found, dpMask, stat,
        resetBuf, resetMax);
}

// ---------------------------------------------------------------------------

int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct { int SM; int Cores; } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
      { 0x20, 32 }, { 0x21, 48 }, { 0x30, 192 }, { 0x32, 192 }, { 0x35, 192 },
      { 0x37, 192 }, { 0x50, 128 }, { 0x52, 128 }, { 0x53, 128 }, { 0x60,  64 },
      { 0x61, 128 }, { 0x62, 128 }, { 0x70,  64 }, { 0x72,  64 }, { 0x75,  64 },
      { 0x80,  64 }, { 0x86, 128 }, { 0x87, 128 }, { 0x89, 128 }, { 0x90, 128 },
      { 0xa0, 128 }, { 0xa1, 128 }, { 0xc0, 128 },   // Ada / Blackwell (RTX 50xx)
      { -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            return nGpuArchCoresPerSM[index].Cores;
        index++;
    }
    return 128;
}

// ---------------------------------------------------------------------------

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
    bool poolMode) {

    this->nbThreadPerGroup = nbThreadPerGroup;
    initialised = false;
    poolDP = poolMode;                 // в pool режим GHT не се заделя (DP се обединяват на хоста)

    inputKangaroo = NULL;
    inputKangarooPinned = NULL;
    outputItem = NULL;
    outputItemPinned = NULL;
    jumpPinned = NULL;
    ghtTable = NULL;
    ghtBytes = 0;
    statCounter = NULL;
    statPinned = NULL;
    resetBuf = NULL;
    resetPinned = NULL;
    jumpXY = NULL;
    jumpCG = NULL;
    scratch = NULL;
    cycBuf = NULL;
    dpBuf = NULL;
    dpMax = 0;
    sassBytes = 0;
    ghtGrid = 0;
    cubinLoaded = false;
    dpLostWarning = false;
    poolDP = false;
    dpMask = 0;

    cudaError_t err;
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        printf("GPUEngine: no CUDA device available\n");
        return;
    }

    err = cudaSetDevice(gpuId);
    if (err != cudaSuccess) {
        printf("GPUEngine: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);

    this->nbThread = nbThreadGroup * nbThreadPerGroup;
    this->maxFound = maxFound;
    this->outputSize = (maxFound * COLL_ITEM_SIZE + 4);

    char tmp[512];
    sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
        gpuId, deviceProp.name, deviceProp.multiProcessorCount,
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
        nbThread / nbThreadPerGroup, nbThreadPerGroup);
    deviceName = std::string(tmp);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // --- състояние на разходките ---
    // Работният набор на разходките (състояние + scratch + cycBuf) се заделя
    // С ЕДИН блок, за да може целият да се сложи в persisting L2 прозорец
    // (както RCKangaroo). Ако не се побира в L2, ядрото чака DRAM.
    kangarooSize = (uint64_t)nbThread * GPU_GRP_SIZE * KSIZE * 8;
    uint64_t perWalk32 = (uint64_t)nbThread * GPU_GRP_SIZE * 32;
    uint64_t kangAligned = (kangarooSize + 255) & ~255ULL;   // STG.128/LDG.128 искат 16 B подравняване
    uint64_t workBytes = kangAligned + 2 * perWalk32;
    err = cudaMalloc((void**)&inputKangaroo, workBytes);
    if (err != cudaSuccess) {
        printf("GPUEngine: walk state alloc failed (%.1f MB): %s\n",
            kangarooSize / 1048576.0, cudaGetErrorString(err));
        return;
    }

    kangarooSizePinned = (uint64_t)nbThreadPerGroup * GPU_GRP_SIZE * KSIZE * 8;
    err = cudaHostAlloc(&inputKangarooPinned, kangarooSizePinned,
        cudaHostAllocWriteCombined | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("GPUEngine: pinned alloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // --- изходен буфер (само колизии) ---
    err = cudaMalloc((void**)&outputItem, outputSize);
    if (err != cudaSuccess) {
        printf("GPUEngine: output alloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaHostAlloc(&outputItemPinned, outputSize, cudaHostAllocMapped);

    // --- таблица със скокове ---
    jumpSize = NB_JUMP * 8 * 4;
    cudaHostAlloc(&jumpPinned, jumpSize, cudaHostAllocMapped);

    // --- статистика: escape / DP / table-full ---
    cudaMalloc((void**)&statCounter, STAT_SLOTS * sizeof(unsigned long long));
    cudaMemset(statCounter, 0, STAT_SLOTS * sizeof(unsigned long long));
    cudaHostAlloc(&statPinned, STAT_SLOTS * sizeof(unsigned long long), cudaHostAllocMapped);

    // --- буфер за рестарт на заклещени разходки ---
    cudaMalloc((void**)&resetBuf, (RESET_MAX + 1) * sizeof(uint32_t));
    cudaMemset(resetBuf, 0, (RESET_MAX + 1) * sizeof(uint32_t));
    cudaHostAlloc(&resetPinned, (RESET_MAX + 1) * sizeof(uint32_t), cudaHostAllocMapped);

    // --- DP таблица в паметта на GPU (само в НЕ-pool режим; в pool режим DP се
    //     обединяват на хоста и GHT не е нужна -> спестяваме ~GBs VRAM + GhtClear) ---
    if (!poolDP) {
        ghtBytes = GHT_SIZE * sizeof(GHT_ENTRY);
        err = cudaMalloc(&ghtTable, ghtBytes);
        if (err != cudaSuccess) {
            printf("GPUEngine: DP table alloc failed (%.2f GB): %s\n"
                "  -> namali GHT_BITS v GPUHashTable.h\n",
                ghtBytes / 1073741824.0, cudaGetErrorString(err));
            return;
        }
        if (!ClearHashTable()) {
            printf("GPUEngine: DP table clear failed\n");
            return;
        }
    }
    else {
        ghtTable = NULL; ghtBytes = 0;
        printf("GPUEngine: pool rezhim -> GHT ne se zadelq (DP se obedinqvat na hosta)\n");
    }

    lostWarning = false;

    // --- буфери на SASS ядрото ---
    // rhoStep адресира с 32-битови отмествания ([R.U32 + UR]) -> всеки буфер < 4 GB.
    // InvMod256 иска пълен warp -> blockDim кратен на 32.
    ghtGrid = deviceProp.multiProcessorCount * 8;
#if RHO_USE_SASS
    if (nbThreadPerGroup % 32 != 0) {
        printf("GPUEngine: SASS rhoStep iska threads/grupa kratno na 32 (sega %d)\n", nbThreadPerGroup);
        return;
    }
    if (kangarooSize >= 0xFFFFFFFFULL) {
        printf("GPUEngine: sastoqnieto na razhodkite e >= 4 GB -> namali grid-a\n");
        return;
    }
#endif
    scratch = (uint64_t*)((uint8_t*)inputKangaroo + kangAligned);
    cycBuf = (uint64_t*)((uint8_t*)inputKangaroo + kangAligned + perWalk32);
    err = cudaMalloc((void**)&jumpXY, (size_t)NB_JUMP * 64);
    if (err == cudaSuccess) err = cudaMalloc((void**)&jumpCG, (size_t)3 * NB_JUMP * 32);
    if (err != cudaSuccess) {
        printf("GPUEngine: SASS buffers alloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemset(cycBuf, 0, perWalk32);     // cyc0/cyc1/cycTot = 0, pend = 0
    sassBytes = (uint64_t)NB_JUMP * 64 + (uint64_t)3 * NB_JUMP * 32 + 2 * perWalk32;

    // --- persisting L2 прозорец ---
    // RHO_L2:  0 = изключен | 1 = САМО състоянието (по подразбиране, като RC) |
    //          2 = целият работен набор (scratch/cyc също -> обикновено по-зле)
    // ВАЖНО: scratch и cycBuf се презаписват всяка стъпка. Да ги закачаме като
    // persisting изяжда L2 запаса и ИЗХВЪРЛЯ реално преизползваното състояние.
    // Затова по подразбиране закачаме само буфера със състоянието на разходките.
    {
        const char* ls = getenv("RHO_L2");
        int mode = ls ? atoi(ls) : 0;   // podrazbirane 0: persisting prozorecat ne pomaga
        size_t l2 = (size_t)deviceProp.l2CacheSize;
        size_t pmax = (size_t)deviceProp.persistingL2CacheMaxSize;
        size_t wmax = (size_t)deviceProp.accessPolicyMaxWindowSize;
        uint64_t pinBytes = (mode == 2) ? workBytes : kangAligned;   // 1 -> само състоянието
        size_t win = (size_t)pinBytes < wmax ? (size_t)pinBytes : wmax;
        size_t pers = win < pmax ? win : pmax;
        printf("GPUEngine: sastoqnie %.1f MB | rabot. nabor %.1f MB | L2 %.1f MB "
            "(persisting max %.1f MB) | RHO_L2=%d%s\n",
            kangAligned / 1048576.0, workBytes / 1048576.0, l2 / 1048576.0,
            pmax / 1048576.0, mode,
            (mode && pinBytes > pmax) ? "  <- prozorecat e po-golqm ot L2, chastichen efekt" : "");
        if (mode != 0 && pers > 0 && win > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pers);
            cudaStreamAttrValue sa;
            memset(&sa, 0, sizeof(sa));
            sa.accessPolicyWindow.base_ptr = inputKangaroo;      // започва от състоянието
            sa.accessPolicyWindow.num_bytes = win;
            sa.accessPolicyWindow.hitRatio = 1.0f;
            sa.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            sa.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &sa);
            cudaGetLastError();   // не е фатално, ако драйверът откаже
        }
    }

#if RHO_USE_SASS
    if (!cubin.LoadCubin("kernel_sm89.cubin")) {
        printf("GPUEngine: cannot load kernel_sm89.cubin\n");
        return;
    }
    cubinLoaded = true;
#endif
    initialised = true;
}

GPUEngine::~GPUEngine() {
    if (inputKangaroo)       cudaFree(inputKangaroo);
    if (outputItem)          cudaFree(outputItem);
    if (ghtTable)            cudaFree(ghtTable);
    if (inputKangarooPinned) cudaFreeHost(inputKangarooPinned);
    if (outputItemPinned)    cudaFreeHost(outputItemPinned);
    if (jumpPinned)          cudaFreeHost(jumpPinned);
    if (statCounter)         cudaFree(statCounter);
    if (statPinned)          cudaFreeHost(statPinned);
    if (resetBuf)            cudaFree(resetBuf);
    if (resetPinned)         cudaFreeHost(resetPinned);
    if (jumpXY)              cudaFree(jumpXY);
    if (jumpCG)              cudaFree(jumpCG);
    if (dpBuf)               cudaFree(dpBuf);
}

// ---------------------------------------------------------------------------

bool GPUEngine::ClearHashTable() {
    GhtClear << <1024, 256 >> > ((GHT_ENTRY*)ghtTable, GHT_SIZE);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("GPUEngine: GhtClear: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// Връща заетата GPU памет в МЕГАБАЙТИ
int GPUEngine::GetMemory() {
    uint64_t dpBytes = dpBuf ? 64 + (uint64_t)dpMax * 64 : 0;
    return (int)((kangarooSize + outputSize + jumpSize + ghtBytes + sassBytes + dpBytes) / 1048576ULL);
}

int GPUEngine::GetGroupSize() { return GPU_GRP_SIZE; }
int GPUEngine::GetNbThread() { return nbThread; }

bool GPUEngine::GetGridSize(int gpuId, int* x, int* y) {
    if (*x <= 0 || *y <= 0) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0 || gpuId >= deviceCount) return false;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, gpuId);
        if (*x <= 0) *x = 2 * deviceProp.multiProcessorCount;
        if (*y <= 0) *y = 128;
    }
    return true;
}

void* GPUEngine::AllocatePinnedMemory(size_t size) {
    void* buff;
    cudaHostAlloc(&buff, size, cudaHostAllocPortable);
    return buff;
}

void GPUEngine::FreePinnedMemory(void* buff) { cudaFreeHost(buff); }

void GPUEngine::PrintCudaInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB)\n",
            i, deviceProp.name, deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            deviceProp.major, deviceProp.minor,
            (double)deviceProp.totalGlobalMem / 1048576.0);
    }
}

// ---------------------------------------------------------------------------
// Layout: 0-3 px | 4-7 py | 8-11 A | 12 state      (KSIZE = 13)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// САМОТЕСТ ОТ ИЗКУСТВЕНА КОЛИЗИЯ (само ако е зададена RHO_TESTKEY).
//
// Ро върви по цялата група, "малък диапазон" няма. Затова тестът е друг:
// взима първата wild разходка с targetIdx = RHO_TESTIDX (по подразбиране 0):
//     R = s * lambda^e * (A*G + T),  T = k*G
// и слага в съседния слот tame разходка в СЪЩАТА точка:
//     A' = A + k (mod n), същите e/neg, b = 0  ->  R = s * lambda^e * A'*G
// Двете вървят еднакво (Ро е детерминистично), удрят една и съща DP ->
// GHT колизия tame/wild -> SolveCollision трябва да върне k.
// Проверява целия път: SASS стъпка, 2-цикли, DP, ghtInsertDP, out, хоста.
// ---------------------------------------------------------------------------
static bool _ParseHex256(const char* s, uint64_t r[4]) {
    r[0] = r[1] = r[2] = r[3] = 0;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s += 2;
    int n = (int)strlen(s);
    if (n == 0 || n > 64) return false;
    for (int i = 0; i < n; i++) {
        char c = s[n - 1 - i];
        int v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return false;
        r[i / 16] |= (uint64_t)v << (4 * (i % 16));
    }
    return true;
}

// r = (x + y) mod n, x,y < n
static void _AddModN(uint64_t r[4], const uint64_t x[4], const uint64_t y[4]) {
    static const uint64_t N[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
                                   0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
    uint64_t t[4], c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s1 = x[i] + y[i];
        uint64_t c1 = s1 < x[i];
        uint64_t s2 = s1 + c;
        uint64_t c2 = s2 < s1;
        t[i] = s2; c = c1 | c2;
    }
    // ако t >= n (или има пренос) -> t -= n
    bool ge = c != 0;
    if (!ge) {
        ge = true;
        for (int i = 3; i >= 0; i--) {
            if (t[i] != N[i]) { ge = t[i] > N[i]; break; }
        }
    }
    if (ge) {
        uint64_t b = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t d1 = t[i] - N[i];
            uint64_t b1 = t[i] < N[i];
            uint64_t d2 = d1 - b;
            uint64_t b2 = d1 < b;
            t[i] = d2; b = b1 | b2;
        }
    }
    for (int i = 0; i < 4; i++) r[i] = t[i];
}

static void _InjectTestCollision(int nbWalk, Int* px, Int* py, Int* a, uint32_t* state) {
    const char* ks = getenv("RHO_TESTKEY");
    if (ks == NULL || ks[0] == 0) return;
    uint64_t k[4];
    if (!_ParseHex256(ks, k)) {
        printf("SELFTEST: RHO_TESTKEY ne e validen hex\n");
        return;
    }
    const char* is = getenv("RHO_TESTIDX");
    uint32_t tIdx = is ? (uint32_t)strtoul(is, NULL, 10) : 0;
    for (int w = 0; w + 1 < nbWalk; w++) {
        if (ST_B(state[w]) == 1 && ST_TIDX(state[w]) == tIdx) {
            int p = w + 1;
            px[p] = px[w];
            py[p] = py[w];
            uint64_t na[4];
            _AddModN(na, a[w].bits64, k);
            for (int i = 0; i < 4; i++) a[p].bits64[i] = na[i];
            a[p].bits64[4] = 0;
            state[p] = state[w] & ~(1u << 27);          // b = 0 (tame), e/neg/tIdx същите
            printf("SELFTEST: tame bliznak na wild razhodka %d (target %u) v slot %d\n", w, tIdx, p);
            return;
        }
    }
    printf("SELFTEST: nqma wild razhodka s target %u -> testat ne e postaven\n", tIdx);
}

void GPUEngine::SetKangaroos(Int* px, Int* py, Int* a, uint32_t* state) {

    int gSize = KSIZE * GPU_GRP_SIZE;
    int strideSize = nbThreadPerGroup * KSIZE;
    int nbBlock = nbThread / nbThreadPerGroup;
    int blockSize = nbThreadPerGroup * gSize;
    int idx = 0;

    _InjectTestCollision(nbThread * GPU_GRP_SIZE, px, py, a, state);

    for (int bl = 0; bl < nbBlock; bl++) {
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            for (int t = 0; t < nbThreadPerGroup; t++) {
                for (int i = 0; i < 4; i++) {
                    inputKangarooPinned[g * strideSize + t + (0 + i) * nbThreadPerGroup] = px[idx].bits64[i];
                    inputKangarooPinned[g * strideSize + t + (4 + i) * nbThreadPerGroup] = py[idx].bits64[i];
                    inputKangarooPinned[g * strideSize + t + (8 + i) * nbThreadPerGroup] = a[idx].bits64[i];
                }
                inputKangarooPinned[g * strideSize + t + 12 * nbThreadPerGroup] = (uint64_t)state[idx];
                idx++;
            }
        }
        uint64_t offset = (uint64_t)bl * blockSize;
        cudaMemcpy(inputKangaroo + offset, inputKangarooPinned,
            kangarooSizePinned, cudaMemcpyHostToDevice);
    }
    // нови разходки -> без 2-цикъл история и без чакащ escape
    if (cycBuf) cudaMemset(cycBuf, 0, (size_t)nbThread * GPU_GRP_SIZE * 32);
}

void GPUEngine::GetKangaroos(Int* px, Int* py, Int* a, uint32_t* state) {

    if (inputKangarooPinned == NULL) return;

    int gSize = KSIZE * GPU_GRP_SIZE;
    int strideSize = nbThreadPerGroup * KSIZE;
    int nbBlock = nbThread / nbThreadPerGroup;
    int blockSize = nbThreadPerGroup * gSize;
    int idx = 0;

    for (int bl = 0; bl < nbBlock; bl++) {
        uint64_t offset = (uint64_t)bl * blockSize;
        cudaMemcpy(inputKangarooPinned, inputKangaroo + offset,
            kangarooSizePinned, cudaMemcpyDeviceToHost);

        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            for (int t = 0; t < nbThreadPerGroup; t++) {
                for (int i = 0; i < 4; i++) {
                    px[idx].bits64[i] = inputKangarooPinned[g * strideSize + t + (0 + i) * nbThreadPerGroup];
                    py[idx].bits64[i] = inputKangarooPinned[g * strideSize + t + (4 + i) * nbThreadPerGroup];
                    a[idx].bits64[i] = inputKangarooPinned[g * strideSize + t + (8 + i) * nbThreadPerGroup];
                }
                px[idx].bits64[4] = 0;
                py[idx].bits64[4] = 0;
                a[idx].bits64[4] = 0;
                state[idx] = (uint32_t)inputKangarooPinned[g * strideSize + t + 12 * nbThreadPerGroup];
                idx++;
            }
        }
    }
}

void GPUEngine::SetKangaroo(uint64_t kIdx, Int* px, Int* py, Int* a, uint32_t state) {

    int gSize = KSIZE * GPU_GRP_SIZE;
    int strideSize = nbThreadPerGroup * KSIZE;
    int blockSize = nbThreadPerGroup * gSize;

    uint64_t t = kIdx % nbThreadPerGroup;
    uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
    uint64_t bl = kIdx / ((uint64_t)nbThreadPerGroup * GPU_GRP_SIZE);

    uint64_t baseOffset = bl * blockSize + g * strideSize + t;

    for (int i = 0; i < 4; i++) {
        inputKangarooPinned[i] = px->bits64[i];
        inputKangarooPinned[4 + i] = py->bits64[i];
        inputKangarooPinned[8 + i] = a->bits64[i];
    }
    inputKangarooPinned[12] = (uint64_t)state;

    for (int i = 0; i < KSIZE; i++) {
        cudaMemcpyAsync(inputKangaroo + baseOffset + (uint64_t)i * nbThreadPerGroup,
            &inputKangarooPinned[i], 8, cudaMemcpyHostToDevice);
    }
    // cycBuf индексът е същият kIdx = (bl*G + g)*blockDim + t: нулираме историята
    // и чакащия escape на рестартираната разходка
    if (cycBuf) cudaMemsetAsync((uint8_t*)cycBuf + kIdx * 32, 0, 32);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

void GPUEngine::SetParams(uint64_t dpMask, Int* jCG_ptr, Int* px, Int* py) {

    this->dpMask = dpMask;

    for (int i = 0; i < NB_JUMP; i++) memcpy(jumpPinned + 4 * i, px[i].bits64, 32);
    cudaMemcpyToSymbol(jPx, jumpPinned, jumpSize);

    for (int i = 0; i < NB_JUMP; i++) memcpy(jumpPinned + 4 * i, py[i].bits64, 32);
    cudaMemcpyToSymbol(jPy, jumpPinned, jumpSize);

    // jCG_ptr e plosak masiv [3 * NB_JUMP]: jCG_ptr[e*NB_JUMP + i]
    for (int e = 0; e < 3; e++) {
        for (int i = 0; i < NB_JUMP; i++)
            memcpy(jumpPinned + 4 * i, jCG_ptr[e * NB_JUMP + i].bits64, 32);
        cudaMemcpyToSymbol(jCG, jumpPinned, jumpSize, (size_t)e * jumpSize);
    }

    // --- същите таблици като глобални буфери за SASS ядрото ---
    // jumpXY[i] = x0..x3 | y0..y3 ; jumpCG[e][i] = jCG[e][i]
    {
        std::vector<uint64_t> h((size_t)NB_JUMP * 8);
        for (int i = 0; i < NB_JUMP; i++) {
            memcpy(&h[(size_t)i * 8], px[i].bits64, 32);
            memcpy(&h[(size_t)i * 8 + 4], py[i].bits64, 32);
        }
        cudaMemcpy(jumpXY, h.data(), (size_t)NB_JUMP * 64, cudaMemcpyHostToDevice);
        std::vector<uint64_t> c((size_t)3 * NB_JUMP * 4);
        for (int e = 0; e < 3; e++)
            for (int i = 0; i < NB_JUMP; i++)
                memcpy(&c[((size_t)e * NB_JUMP + i) * 4], jCG_ptr[e * NB_JUMP + i].bits64, 32);
        cudaMemcpy(jumpCG, c.data(), (size_t)3 * NB_JUMP * 32, cudaMemcpyHostToDevice);
    }

    // --- DP буфер: 2 x очакваните DP за едно извикване + запас ---
    {
        int dpBits = 0;
        for (uint64_t m = dpMask; m; m >>= 1) dpBits += (int)(m & 1);
        double expDP = (double)nbThread * GPU_GRP_SIZE * NB_RUN / ldexp(1.0, dpBits);
        double want = 2.0 * expDP + 65536.0;
        const double capDP = (double)(1u << 22);          // 4M записа = 256 MB
        if (want > capDP) {
            printf("GPUEngine: ~%.0f DP na izvikvane > %.0f -> DP shte se gubqt, uvelichi -d\n",
                expDP, capDP);
            want = capDP;
        }
        uint32_t newMax = (uint32_t)want;
        if (dpBuf == NULL || newMax != dpMax) {
            if (dpBuf) cudaFree(dpBuf);
            dpBuf = NULL;
            dpMax = newMax;
            if (cudaMalloc((void**)&dpBuf, 64 + (size_t)dpMax * 64) != cudaSuccess) {
                printf("GPUEngine: dpBuf alloc failed (%u zapisa)\n", dpMax);
                dpBuf = NULL; dpMax = 0; initialised = false;
                return;
            }
            cudaMemset(dpBuf, 0, 64);
        }
    }
}

// ---------------------------------------------------------------------------

bool GPUEngine::callKernel() {
    if (!initialised) {
        printf("GPUEngine: not initialised\n");
        return false;
    }
    cudaMemset(outputItem, 0, 4);
    cudaMemset(resetBuf, 0, 4);

#if RHO_USE_SASS
    if (!dpBuf) {
        printf("GPUEngine: SetParams() ne e izvikan (nqma dpBuf)\n");
        return false;
    }
    cudaMemset(dpBuf, 0, 4);

    RhoSassParams kp;
    memset(&kp, 0, sizeof(kp));
    kp.kangaroos = inputKangaroo;
    kp.ght = ghtTable;
    kp.out = outputItem;
    kp.stat = statCounter;
    kp.resetBuf = resetBuf;
    kp.jumpXY = jumpXY;
    kp.jumpCG = jumpCG;
    kp.scratch = scratch;
    kp.dpMask = dpMask;
    kp.maxFound = maxFound;
    kp.resetMax = RESET_MAX;
    kp.nbRun = NB_RUN;
    kp.grpSize = GPU_GRP_SIZE;
    kp.dpBuf = dpBuf;
    kp.dpMax = dpMax;
    kp.cycBuf = cycBuf;

    TCallKernelParams p;
    memset(&p, 0, sizeof(p));
    strcpy(p.kernel_name, "rhoStep");
    p.blockSize = nbThreadPerGroup;
    p.blockCnt = nbThread / nbThreadPerGroup;
    // dinamichna shared pamet za jump tablicata: jumpXY (256*64) + jumpCG (3*256*32)
    p.sharedSize = NB_JUMP * 64 + 3 * NB_JUMP * 32;   // = 40960 B za NB_JUMP=256
    p.stream = 0;
    p.kernel_param_ptr = &kp;
    p.kernel_param_size = sizeof(kp);
    if (!cubin.CallKernel(p)) return false;

    // В pool режим НЕ вмъкваме в GHT -- DP се обединяват на хоста (между картите).
    if (!poolDP) {
        // същия (default) stream -> тръгва след rhoStep
        ghtInsertDP << <ghtGrid, 256 >> > (dpBuf, (GHT_ENTRY*)ghtTable, outputItem,
            statCounter, dpMax, maxFound);
    }
#else
    comp_kangaroos << <nbThread / nbThreadPerGroup, nbThreadPerGroup >> > (
        inputKangaroo, (GHT_ENTRY*)ghtTable, maxFound, outputItem, dpMask,
        statCounter, resetBuf, RESET_MAX);
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool GPUEngine::callKernelAndWait() {
    if (!callKernel()) return false;
    cudaMemcpy(outputItemPinned, outputItem, outputSize, cudaMemcpyDeviceToHost);
    return cudaGetLastError() == cudaSuccess;
}

bool GPUEngine::Launch(std::vector<COLLISION>& collFound, bool spinWait) {

    collFound.clear();

    cudaError_t serr = cudaDeviceSynchronize();
    if (serr != cudaSuccess) {
        printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(serr));
        return false;
    }
    cudaMemcpy(outputItemPinned, outputItem, 4, cudaMemcpyDeviceToHost);

#if RHO_USE_SASS
    if (!dpLostWarning && dpBuf) {
        uint32_t dpCnt = 0;
        cudaMemcpy(&dpCnt, dpBuf, 4, cudaMemcpyDeviceToHost);
        if (dpCnt > dpMax) {
            printf("\nWarning: %u DP izgubeni (dpBuf %u zapisa pravi) -> uvelichi -d\n",
                dpCnt - dpMax, dpMax);
            dpLostWarning = true;
        }
    }
#endif

    uint32_t nb = outputItemPinned[0];
    if (nb > maxFound) {
        if (!lostWarning) {
            printf("\nWarning: %u collisions lost (buffer full)\n", nb - maxFound);
            lostWarning = true;
        }
        nb = maxFound;
    }

    if (nb > 0) {
        printf("\n>>> %u COLLISION(S) FROM GPU\n", nb);
        cudaMemcpy(outputItemPinned + 1, outputItem + 1,
            (size_t)nb * COLL_ITEM_SIZE, cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i < nb; i++) {
            uint32_t* p = outputItemPinned + 1 + i * COLL_ITEM_SIZE32;
            COLLISION c;
            memcpy(c.x, p + 0, 16);
            memcpy(c.a1, p + 4, 32);
            c.st1 = p[12];
            memcpy(c.a2, p + 13, 32);
            c.st2 = p[21];
            collFound.push_back(c);
        }
    }

    return callKernel();
}

// ---------------------------------------------------------------------------
// Много-GPU: взима DP от този пуск (без вмъкване в GHT) и пуска пак.
// Викай със SetPoolMode(true). Хостът обединява DP и лови колизии между картите.
// ---------------------------------------------------------------------------
bool GPUEngine::LaunchPool(std::vector<GpuDP>& out) {
    out.clear();

    cudaError_t serr = cudaDeviceSynchronize();
    if (serr != cudaSuccess) {
        printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(serr));
        return false;
    }
#if RHO_USE_SASS
    if (dpBuf) {
        uint32_t dpCnt = 0;
        cudaMemcpy(&dpCnt, dpBuf, 4, cudaMemcpyDeviceToHost);
        uint32_t n = dpCnt < dpMax ? dpCnt : dpMax;
        if (dpCnt > dpMax && !dpLostWarning) {
            printf("\nWarning: %u DP izgubeni (dpBuf %u zapisa) -> uvelichi -d\n",
                dpCnt - dpMax, dpMax);
            dpLostWarning = true;
        }
        if (n > 0) {
            // запис i на байт 64 + 64*i: x0 x1 a0 a1 a2 a3 st pad  (8 x u64)
            std::vector<uint64_t> raw((size_t)n * 8);
            cudaMemcpy(raw.data(), (uint8_t*)dpBuf + 64,
                (size_t)n * 64, cudaMemcpyDeviceToHost);
            out.reserve(n);
            for (uint32_t i = 0; i < n; i++) {
                const uint64_t* r = &raw[(size_t)i * 8];
                GpuDP d;
                d.x[0] = r[0]; d.x[1] = r[1];
                d.a[0] = r[2]; d.a[1] = r[3]; d.a[2] = r[4]; d.a[3] = r[5];
                d.st = (uint32_t)r[6];
                out.push_back(d);
            }
        }
    }
#endif
    return callKernel();
}

// ---------------------------------------------------------------------------
// Статистика: кумулативни броячи от старта на процеса.
//   escape -- засечени безплодни цикли
//   dp     -- произведени distinguished points
//   full   -- DP точки, отхвърлени заради препълнена кофа
// ---------------------------------------------------------------------------
void GPUEngine::GetStats(uint64_t* out) {
    if (statCounter == NULL) {
        for (int i = 0; i < STAT_SLOTS; i++) out[i] = 0;
        return;
    }
    cudaMemcpy(statPinned, statCounter, STAT_SLOTS * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost);
    for (int i = 0; i < STAT_SLOTS; i++) out[i] = (uint64_t)statPinned[i];
}

// ---------------------------------------------------------------------------
// Индекси на разходките, които ядрото е маркирало като заклещени.
// Викай СЛЕД Launch и ПРЕДИ следващото callKernel.
// ---------------------------------------------------------------------------
uint32_t GPUEngine::GetResetList(std::vector<uint32_t>& idx) {
    idx.clear();
    if (resetBuf == NULL) return 0;

    cudaMemcpy(resetPinned, resetBuf, 4, cudaMemcpyDeviceToHost);
    uint32_t n = resetPinned[0];
    if (n == 0) return 0;
    if (n > RESET_MAX) n = RESET_MAX;

    cudaMemcpy(resetPinned + 1, resetBuf + 1, n * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < n; i++) idx.push_back(resetPinned[1 + i]);
    return n;
}
