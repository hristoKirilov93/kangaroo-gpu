#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include <string>
#include "../Constants.h"
#include "../SECPK1/SECP256k1.h"
#include "../CallCubin.h"      // RC: TCubinCall (същият include като досега)

// ---------------------------------------------------------------------------
// Избор на ядро:
//   1 = SASS rhoStep (kernel_sm89.cubin) + ghtInsertDP (.cu)
//   0 = старото .cu ядро comp_kangaroos (референция за сравнение)
// ---------------------------------------------------------------------------
#ifndef RHO_USE_SASS
#define RHO_USE_SASS 1
#endif

// ---------------------------------------------------------------------------
// РО РЕЖИМ: състояние на разходка = px(4) + py(4) + A(4) + state(1) = 13
// ---------------------------------------------------------------------------
#define KSIZE 13

// ---------------------------------------------------------------------------
// Диагностични броячи. Layout:
//   0 cycle | 1 DP | 2 full | 10 escape | 14 reset   (пишат ги и двете ядра)
//   останалите -- само старата диагностика на .cu ядрото
// ---------------------------------------------------------------------------
#define STAT_HIST_BASE 24
#define STAT_SLOTS     (STAT_HIST_BASE + NB_JUMP)

// Максимален брой разходки, рестартирани за едно извикване на ядрото
#define RESET_MAX 8192

// ---------------------------------------------------------------------------
// Изходен елемент = КОЛИЗИЯ, не DP.
//   x префикс      : 4 x uint32
//   A1 + state1    : 8 + 1
//   A2 + state2    : 8 + 1
// Общо 22 x uint32 = 88 байта
// ---------------------------------------------------------------------------
#define COLL_ITEM_SIZE32 22
#define COLL_ITEM_SIZE   (COLL_ITEM_SIZE32 * 4)

// Съвместимост със стария код
#define ITEM_SIZE   COLL_ITEM_SIZE
#define ITEM_SIZE32 COLL_ITEM_SIZE32

#pragma pack(push, 1)
typedef struct {
    uint64_t x[2];     // 128-битов префикс на x
    uint64_t a1[4];    // скалар на разходка 1 (нескориган за GLV)
    uint32_t st1;      // e | neg | b | tIdx
    uint64_t a2[4];    // скалар на разходка 2
    uint32_t st2;
} COLLISION;

// Оставено за съвместимост със стар код, който още реферира ITEM
typedef COLLISION ITEM;
#pragma pack(pop)

// Разопаковане на state полето
#define ST_E(s)     (((s) >> 29) & 3u)
#define ST_NEG(s)   (((s) >> 28) & 1u)
#define ST_B(s)     (((s) >> 27) & 1u)
#define ST_TIDX(s)  ((s) & 0x07FFFFFFu)
#define ST_PACK(e, neg, b, tidx) \
    ((((uint32_t)(e) & 3u) << 29) | (((uint32_t)(neg) & 1u) << 28) | \
     (((uint32_t)(b) & 1u) << 27) | ((uint32_t)(tidx) & 0x07FFFFFFu))

class GPUEngine {
public:
    GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,
        bool poolMode = false);
    ~GPUEngine();

    // jCG_ptr: [3][NB_JUMP] предварително мащабирани скалари lambda^-e * c_i
    void SetParams(uint64_t dpMask, Int* jCG_ptr, Int* px, Int* py);

    void SetKangaroos(Int* px, Int* py, Int* a, uint32_t* state);
    void GetKangaroos(Int* px, Int* py, Int* a, uint32_t* state);
    void SetKangaroo(uint64_t kIdx, Int* px, Int* py, Int* a, uint32_t state);

    bool Launch(std::vector<COLLISION>& collFound, bool spinWait = false);
    bool ClearHashTable();
    uint32_t GetResetList(std::vector<uint32_t>& idx);

    // --- mnogo-GPU: obedineni DP na hosta ---
    // Pri poolMode=true qdroto NE pravi vmakvane v GHT (ghtInsertDP ne se puska);
    // vmesto tova DP se vzimat na hosta s GetDPs i vlizat v obshta hash tablica,
    // taka che koliziite se lovqt MEZHDU vsichki karti.
    struct GpuDP { uint64_t x[2]; uint64_t a[4]; uint32_t st; };
    void SetPoolMode(bool on) { poolDP = on; }
    // puska qdroto (kakto Launch), sinhronizira i vrashta DP ot tozi pusk v out.
    // vrashta broya DP (ogranichen do dpMax).
    bool LaunchPool(std::vector<GpuDP>& out);
    void GetStats(uint64_t* out);    // буферът трябва да е с размер STAT_SLOTS

    int GetNbThread();
    int GetGroupSize();
    int GetMemory();
    bool callKernelAndWait();
    bool callKernel();

    std::string deviceName;

    static void* AllocatePinnedMemory(size_t size);
    static void FreePinnedMemory(void* buff);
    static void PrintCudaInfo();
    static bool GetGridSize(int gpuId, int* x, int* y);

private:
    uint64_t* inputKangaroo;
    uint64_t* inputKangarooPinned;
    uint32_t* outputItem;
    uint32_t* outputItemPinned;
    uint64_t* jumpPinned;

    void* ghtTable;        // GHT_ENTRY* -- DP таблицата в GPU паметта
    uint64_t  ghtBytes;

    unsigned long long* statCounter;
    unsigned long long* statPinned;

    uint32_t* resetBuf;        // [0] = брояч, [1..] = индекси на разходки
    uint32_t* resetPinned;

    // --- буфери на SASS ядрото ---
    uint64_t* jumpXY;      // [NB_JUMP][x0..x3 | y0..y3]
    uint64_t* jumpCG;      // [3][NB_JUMP][4]
    uint64_t* scratch;     // префиксни произведения: nbThread * GPU_GRP_SIZE * 32 B
    uint64_t* cycBuf;      // 2-цикъл състояние: nbThread * GPU_GRP_SIZE * 32 B
    uint64_t* dpBuf;       // [0] брояч, записи по 64 B от байт 64
    uint32_t  dpMax;
    uint64_t  sassBytes;
    int       ghtGrid;
    TCubinCall cubin;
    bool cubinLoaded;
    bool dpLostWarning;
    bool poolDP;

    uint64_t dpMask;
    int nbThread;
    int nbThreadPerGroup;
    uint32_t maxFound;
    uint32_t outputSize;
    uint64_t kangarooSize;
    uint64_t kangarooSizePinned;
    uint32_t jumpSize;
    bool initialised;
    bool lostWarning;
};

#endif // GPUENGINEH
