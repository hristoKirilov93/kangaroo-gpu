#include "HashTable.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#ifndef WIN64
#include <sys/stat.h>
#include <unistd.h>
#endif

HashTable::HashTable() {
    memset(E, 0, sizeof(E));
}

void HashTable::Reset() {
    for (uint32_t h = 0; h < HASH_SIZE; h++) {
        if (E[h].items) {
            free(E[h].items);
            E[h].items = NULL;
        }
        E[h].maxItem = 0;
        E[h].nbItem = 0;
    }
}

uint64_t HashTable::GetNbItem() {
    uint64_t totalItem = 0;
    for (uint64_t h = 0; h < HASH_SIZE; h++)
        totalItem += (uint64_t)E[h].nbItem;
    return totalItem;
}

void HashTable::Convert(Int* x, Int* a, Int* b, uint32_t type, uint64_t* h, ENTRY* e) {
    e->x[0] = x->bits64[0];
    e->x[1] = x->bits64[1];
    e->x[2] = x->bits64[2];
    e->x[3] = x->bits64[3];

    e->a[0] = a->bits64[0];
    e->a[1] = a->bits64[1];
    e->a[2] = a->bits64[2];
    e->a[3] = a->bits64[3];

    e->b[0] = b->bits64[0];
    e->b[1] = b->bits64[1];
    e->b[2] = b->bits64[2];
    e->b[3] = b->bits64[3];

    e->type = type;

    uint64_t shifted = (x->bits64[0] >> 16) ^ x->bits64[1] ^ x->bits64[2] ^ x->bits64[3];
    *h = (shifted & HASH_MASK);
}

int HashTable::Add(Int* x, Int* a, Int* b, uint32_t type) {
    ENTRY e;
    uint64_t h;
    Convert(x, a, b, type, &h, &e);
    return Add(h, &e);
}

void HashTable::ReAllocate(uint64_t h, uint32_t add) {
    uint32_t newMax = (E[h].maxItem == 0) ? 16 : (E[h].maxItem * 2 + add);

    ENTRY* nitems = (ENTRY*)realloc(E[h].items, sizeof(ENTRY) * newMax);
    if (nitems == NULL) {
        printf("Fatal Error: Out of memory in HashTable::ReAllocate\n");
        exit(-1);
    }
    E[h].items = nitems;
    E[h].maxItem = newMax;
}

// ------------------------------------------------------------------
// DP на диск
// ------------------------------------------------------------------

std::string HashTable::BucketFileName(const std::string& dir, uint64_t h) {
    char buf[32];
    sprintf(buf, "%06llx.bin", (unsigned long long)h);
    return dir + "/" + std::string(buf);
}

// Двоично търсене директно във файла (сортиран), без да го зарежда изцяло
int HashTable::DiskSearch(const std::string& fileName, ENTRY* e, ENTRY* foundEntry) {
    FILE* f = fopen(fileName.c_str(), "rb");
    if (!f) return ADD_OK;

#ifdef WIN64
    _fseeki64(f, 0, SEEK_END);
    uint64_t fileSize = (uint64_t)_ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    uint64_t fileSize = (uint64_t)ftello(f);
#endif
    int64_t nbItem = (int64_t)(fileSize / sizeof(ENTRY));

    int64_t st = 0, ed = nbItem - 1;
    while (st <= ed) {
        int64_t mi = (st + ed) / 2;
#ifdef WIN64
        _fseeki64(f, mi * (int64_t)sizeof(ENTRY), SEEK_SET);
#else
        fseeko(f, mi * (int64_t)sizeof(ENTRY), SEEK_SET);
#endif
        ENTRY cur;
        if (fread(&cur, sizeof(ENTRY), 1, f) != 1) break;

        int comp = compare(e->x, cur.x);
        if (comp < 0) {
            ed = mi - 1;
        }
        else if (comp == 0) {
            *foundEntry = cur;
            fclose(f);
            if (e->a[0] == cur.a[0] && e->a[1] == cur.a[1] && e->a[2] == cur.a[2] && e->a[3] == cur.a[3] &&
                e->b[0] == cur.b[0] && e->b[1] == cur.b[1] && e->b[2] == cur.b[2] && e->b[3] == cur.b[3]) {
                return ADD_DUPLICATE;
            }
            return ADD_COLLISION;
        }
        else {
            st = mi + 1;
        }
    }
    fclose(f);
    return ADD_OK;
}

int HashTable::Add(uint64_t h, ENTRY* e) {
    // 1. Проверка в RAM бъкета
    if (E[h].items != NULL && E[h].nbItem > 0) {
        int st = 0, ed = (int)E[h].nbItem - 1, mi = 0;
        while (st <= ed) {
            mi = (st + ed) / 2;
            int comp = compare(e->x, E[h].items[mi].x);
            if (comp < 0) {
                ed = mi - 1;
            }
            else if (comp == 0) {
                if (e->a[0] == E[h].items[mi].a[0] && e->a[1] == E[h].items[mi].a[1] &&
                    e->a[2] == E[h].items[mi].a[2] && e->a[3] == E[h].items[mi].a[3] &&
                    e->b[0] == E[h].items[mi].b[0] && e->b[1] == E[h].items[mi].b[1] &&
                    e->b[2] == E[h].items[mi].b[2] && e->b[3] == E[h].items[mi].b[3]) {
                    return ADD_DUPLICATE;
                }
                kA.SetInt32(0); kB.SetInt32(0);
                kA.bits64[0] = E[h].items[mi].a[0]; kA.bits64[1] = E[h].items[mi].a[1];
                kA.bits64[2] = E[h].items[mi].a[2]; kA.bits64[3] = E[h].items[mi].a[3];
                kB.bits64[0] = E[h].items[mi].b[0]; kB.bits64[1] = E[h].items[mi].b[1];
                kB.bits64[2] = E[h].items[mi].b[2]; kB.bits64[3] = E[h].items[mi].b[3];
                kType = E[h].items[mi].type;
                return ADD_COLLISION;
            }
            else {
                st = mi + 1;
            }
        }
    }

    // 2. Проверка на диск (само ако е зададена директория)
    if (!diskDir.empty()) {
        ENTRY found;
        std::string fn = BucketFileName(diskDir, h);
        int diskRet = DiskSearch(fn, e, &found);

        if (diskRet == ADD_DUPLICATE) return ADD_DUPLICATE;

        if (diskRet == ADD_COLLISION) {
            kA.SetInt32(0); kB.SetInt32(0);
            kA.bits64[0] = found.a[0]; kA.bits64[1] = found.a[1];
            kA.bits64[2] = found.a[2]; kA.bits64[3] = found.a[3];
            kB.bits64[0] = found.b[0]; kB.bits64[1] = found.b[1];
            kB.bits64[2] = found.b[2]; kB.bits64[3] = found.b[3];
            kType = found.type;
            return ADD_COLLISION;
        }
    }

    // 3. Вмъкване в RAM (сортирано)
    if (E[h].items == NULL || E[h].nbItem >= E[h].maxItem) {
        ReAllocate(h, 16);
    }

    if (E[h].nbItem == 0) {
        E[h].items[0] = *e;
        E[h].nbItem = 1;
        return ADD_OK;
    }

    int st = 0, ed = (int)E[h].nbItem - 1, mi = 0;
    while (st <= ed) {
        mi = (st + ed) / 2;
        int comp = compare(e->x, E[h].items[mi].x);
        if (comp < 0) ed = mi - 1;
        else st = mi + 1;
    }

    int numToMove = (int)E[h].nbItem - st;
    if (numToMove > 0) {
        memmove(&E[h].items[st + 1], &E[h].items[st], sizeof(ENTRY) * numToMove);
    }
    E[h].items[st] = *e;
    E[h].nbItem++;

    return ADD_OK;
}

std::string HashTable::GetSizeInfo() {
    return std::string("");
}

void HashTable::PrintInfo() {}