#ifndef GPUHASHTABLEH
#define GPUHASHTABLEH

// ---------------------------------------------------------------------------
// DP таблица изцяло в паметта на GPU.
//
// Заменя дисковия backend. Няма PCIe трафик за всяка DP точка, няма
// LOCK(ghMutex) на хоста, няма I/O паузи. CPU-то вижда САМО реални колизии.
//
// Структура: open addressing с линейно пробване и atomicCAS.
// Ключ: първите 128 бита на x-координатата (след канонизация).
//
// Размер (GHT_ENTRY = 56 байта):
//   GHT_BITS = 26 -> 67M слота x 56 байта = 3.5 GB
//   GHT_BITS = 25 -> 33M слота x 56 байта = 1.75 GB
//
// ПОПРАВКА НА RACE (проверена на GPU в 3.6): оригиналът четеше meta и a0..a3
// с обикновени loads (през некохерентния L1, без acquire). При две едновременни
// вмъквания на една и съща точка читателят виждаше VALID, но стари a0..a3
// (невалидни колизии), или въртеше на стара стойност и колизията се губеше.
// Сега: meta се чете атомарно (L2), публикуване с atomicExch, след VALID ->
// __threadfence() (acquire), товарът се чете с __ldcg (само L2).
// ---------------------------------------------------------------------------

#include <stdint.h>
#include <cuda_runtime.h>

#ifndef GHT_BITS
#define GHT_BITS 26
#endif

#define GHT_SIZE  (1ULL << GHT_BITS)
#define GHT_MASK  (GHT_SIZE - 1ULL)
#define GHT_EMPTY 0xFFFFFFFFFFFFFFFFULL
#define GHT_PROBE 24

#define GHT_INSERTED  0
#define GHT_DUPLICATE 1
#define GHT_COLLISION 2
#define GHT_FULL      3

// meta layout:
//   bit  31      : valid flag (записът е напълно записан)
//   bits 29-30   : e  (GLV степен 0/1/2)
//   bit  28      : neg (приложено отрицание)
//   bit  27      : b   (0 = tame, 1 = wild)
//   bits 0-26    : targetIdx (до 134M таргета)
#define GHT_VALID_BIT 0x80000000u

#pragma pack(push, 1)
typedef struct {
    uint64_t x0;      // младши 64 бита на x
    uint64_t x1;      // следващи 64 бита на x
    uint64_t a0, a1, a2, a3;  // натрупаният скалар A (нескориган)
    uint32_t meta;
    uint32_t pad;
} GHT_ENTRY;          // 56 байта
#pragma pack(pop)

// ---------------------------------------------------------------------------
// splitmix64 -- бърз и добре разпределен
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t _GhtHash(uint64_t x0, uint64_t x1) {
    uint64_t h = x0 ^ (x1 * 0x9E3779B97F4A7C15ULL);
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

__device__ __forceinline__ uint64_t _GhtLd(const uint64_t* p) {
    return (uint64_t)__ldcg((const unsigned long long*)p);    // L2, без L1
}
__device__ __forceinline__ uint32_t _GhtLdMeta(uint32_t* p) {
    return atomicAdd(p, 0u);                                   // кохерентно четене
}

// ---------------------------------------------------------------------------
// Вмъкване с откриване на колизия.
//
// Връща:
//   GHT_INSERTED  -- нов запис, продължавай
//   GHT_DUPLICATE -- същата точка, същите скалари (същата нишка е минала пак)
//   GHT_COLLISION -- същата точка, РАЗЛИЧНИ скалари -> *found съдържа другия
//   GHT_FULL      -- кофата е пълна, записът се изхвърля
// ---------------------------------------------------------------------------
__device__ int GhtInsert(GHT_ENTRY* table,
                         const uint64_t* x, const uint64_t* a,
                         uint32_t meta, GHT_ENTRY* found) {

    uint64_t h = _GhtHash(x[0], x[1]);

    for (int probe = 0; probe < GHT_PROBE; probe++) {

        uint64_t slot = (h + (uint64_t)probe) & GHT_MASK;
        uint64_t cur = _GhtLd(&table[slot].x0);

        if (cur == GHT_EMPTY) {
            uint64_t old = atomicCAS((unsigned long long*)&table[slot].x0,
                                     (unsigned long long)GHT_EMPTY,
                                     (unsigned long long)x[0]);
            if (old == GHT_EMPTY) {
                // Спечелихме слота -- дописваме останалото
                table[slot].x1 = x[1];
                table[slot].a0 = a[0];
                table[slot].a1 = a[1];
                table[slot].a2 = a[2];
                table[slot].a3 = a[3];
                __threadfence();                                        // release
                atomicExch(&table[slot].meta, meta | GHT_VALID_BIT);    // публикуване
                return GHT_INSERTED;
            }
            cur = old;  // друга нишка ни изпревари -- проверяваме нейния запис
        }

        if (cur != x[0]) continue;

        // Изчакваме записът да е напълно готов (кохерентно, с отстъпка)
        uint32_t m = _GhtLdMeta(&table[slot].meta);
        int spin = 0;
        while ((m & GHT_VALID_BIT) == 0 && spin < 100000) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
            __nanosleep(64);
#endif
            m = _GhtLdMeta(&table[slot].meta);
            spin++;
        }
        if ((m & GHT_VALID_BIT) == 0) continue;  // изоставяме, много рядко
        __threadfence();                          // acquire: товарът се чете СЛЕД VALID

        if (_GhtLd(&table[slot].x1) != x[1]) continue;  // само x0 съвпада

        uint64_t t0 = _GhtLd(&table[slot].a0);
        uint64_t t1 = _GhtLd(&table[slot].a1);
        uint64_t t2 = _GhtLd(&table[slot].a2);
        uint64_t t3 = _GhtLd(&table[slot].a3);

        // Същата точка. Дубликат ли е или реална колизия?
        if (t0 == a[0] && t1 == a[1] && t2 == a[2] && t3 == a[3] &&
            (m & ~GHT_VALID_BIT) == meta) {
            return GHT_DUPLICATE;
        }

        found->x0 = x[0];
        found->x1 = x[1];
        found->a0 = t0;
        found->a1 = t1;
        found->a2 = t2;
        found->a3 = t3;
        found->meta = m & ~GHT_VALID_BIT;
        return GHT_COLLISION;
    }

    return GHT_FULL;
}

// ---------------------------------------------------------------------------
// Инициализация -- всички слотове празни
// ---------------------------------------------------------------------------
__global__ void GhtClear(GHT_ENTRY* table, uint64_t nbSlot) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; i < nbSlot; i += stride) {
        table[i].x0 = GHT_EMPTY;
        table[i].meta = 0;
    }
}

#endif // GPUHASHTABLEH
