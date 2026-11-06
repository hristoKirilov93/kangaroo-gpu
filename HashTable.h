#ifndef HASHTABLEH
#define HASHTABLEH

#include <string>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include "SECPK1/Point.h"

#ifdef WIN64
#include <Windows.h>
#include <direct.h>
#endif

#define HASH_SIZE_BIT 18
#define HASH_SIZE (1 << HASH_SIZE_BIT)
#define HASH_MASK (HASH_SIZE - 1)

#define ADD_OK        0
#define ADD_DUPLICATE 1
#define ADD_COLLISION 2

#pragma pack(push, 1)
union int128_s {
    uint8_t  i8[16];
    uint16_t i16[8];
    uint32_t i32[4];
    uint64_t i64[2];
};
typedef union int128_s int128_t;

typedef struct {
    uint64_t x[4];
    uint64_t a[4];
    uint64_t b[4];
    uint32_t type;
} ENTRY;
#pragma pack(pop)

#define safe_free(x) if(x) { free(x); x = NULL; }

typedef struct {
    uint32_t   nbItem;
    uint32_t   maxItem;
    ENTRY* items;
} HASH_ENTRY;

class HashTable {
public:
    HashTable();
    int Add(Int* x, Int* a, Int* b, uint32_t type);
    int Add(uint64_t h, ENTRY* e);
    uint64_t GetNbItem();
    void Reset();
    std::string GetSizeInfo();
    void PrintInfo();

    void SaveTable(FILE* f) {}
    void SaveTable(FILE* f, uint32_t from, uint32_t to, bool printPoint = true) {}
    void LoadTable(FILE* f) {}
    void LoadTable(FILE* f, uint32_t from, uint32_t to) {}
    void SeekNbItem(FILE* f, bool restorePos = false) {}
    void SeekNbItem(FILE* f, uint32_t from, uint32_t to) {}

    void ReAllocate(uint64_t h, uint32_t add);

    HASH_ENTRY E[HASH_SIZE];

    Int      kA;
    Int      kB;
    uint32_t kType;

    static void Convert(Int* x, Int* a, Int* b, uint32_t type, uint64_t* h, ENTRY* e);

private:
    std::string diskDir;

    static __forceinline int compare(const uint64_t a[4], const uint64_t b[4]) {
        if (a[3] != b[3]) return (a[3] > b[3]) ? 1 : -1;
        if (a[2] != b[2]) return (a[2] > b[2]) ? 1 : -1;
        if (a[1] != b[1]) return (a[1] > b[1]) ? 1 : -1;
        if (a[0] != b[0]) return (a[0] > b[0]) ? 1 : -1;
        return 0;
    }

    static std::string BucketFileName(const std::string& dir, uint64_t h);
    int DiskSearch(const std::string& fileName, ENTRY* e, ENTRY* foundEntry);
};

#endif // HASHTABLEH