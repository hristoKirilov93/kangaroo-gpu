/*
 * Based on the Kangaroo distribution by Jean Luc PONS.
 * Modified: Pollard's Rho, multi-target, GLV endomorphism,
 *           GPU-resident DP table.
 */

#include <cstdlib>
#include <cstring>

#ifndef POOL_BITS
#define POOL_BITS 28            // 2^23 = 8.4M записа (~448 MB). Смени според RAM.
#endif

#include "Kangaroo.h"
#include "SHA256.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#endif

using namespace std;

#define safe_delete_array(x) if(x) {delete[] x;x=NULL;}

// ---------------------------------------------------------------------------
// Конструктор
// ---------------------------------------------------------------------------
Kangaroo::Kangaroo(Secp256K1* secp, int32_t initDPSize, bool useGpu,
    string& workFile, string& iWorkFile, uint32_t savePeriod, bool saveKangaroo,
    bool saveKangarooByServer, double maxStep, int wtimeout, int port,
    int ntimeout, string serverIp, string outputFile, bool splitWorkfile) {

    this->secp = secp;
    this->initDPSize = initDPSize;
    this->useGpu = useGpu;
    this->offsetCount = 0;
    this->offsetTime = 0.0;
    this->workFile = workFile;
    this->saveWorkPeriod = savePeriod;
    this->inputFile = iWorkFile;
    this->nbLoadedWalk = 0;
    this->clientMode = serverIp.length() > 0;
    this->saveKangarooByServer = this->clientMode && saveKangarooByServer;
    this->saveKangaroo = saveKangaroo || this->saveKangarooByServer;
    this->fRead = NULL;
    this->maxStep = maxStep;
    this->wtimeout = wtimeout;
    this->port = port;
    this->ntimeout = ntimeout;
    this->serverIp = serverIp;
    this->outputFile = outputFile;
    this->hostInfo = NULL;
    this->endOfSearch = false;
    this->saveRequest = false;
    this->connectedClient = 0;
    this->totalRW = 0;
    this->collisionInSameHerd = 0;
    this->keyIdx = 0;
    this->splitWorkfile = splitWorkfile;
    this->pid = Timer::getPID();

    CPU_GRP_SIZE = 1024;

    InitGLV();

#ifdef WIN64
    ghMutex = CreateMutex(NULL, FALSE, NULL);
    saveMutex = CreateMutex(NULL, FALSE, NULL);
#else
    pthread_mutex_init(&ghMutex, NULL);
    pthread_mutex_init(&saveMutex, NULL);
    signal(SIGPIPE, SIG_IGN);
#endif
}

// ---------------------------------------------------------------------------
// GLV константи
//   lambda^3 = 1 mod n,  beta^3 = 1 mod p,  phi(x,y) = (beta*x, y) = lambda*P
//   Полезно: lambda^-1 == lambda^2  и  lambda^-2 == lambda
// ---------------------------------------------------------------------------
void Kangaroo::InitGLV() {
    LAMBDA.SetBase16((char*)"5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
    LAMBDA2.SetBase16((char*)"AC9C52B33FA3CF1F5AD9E3FD77ED9BA4A880B9FC8EC739C2E0CFC810B51283CE");
    BETA_H.SetBase16((char*)"7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE");
    BETA2_H.SetBase16((char*)"851695D49A83F8EF919BB86153CBCB16630FB68AED0A766A3EC693D68E6AFA40");
    PHALF_H.SetBase16((char*)"7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE17");
}

// ---------------------------------------------------------------------------
// Каноничен представител на 6-орния клас { +-P, +-phi(P), +-phi^2(P) }
// ТРЯБВА да е идентична с _CanonicalClass в GPUGLV.h
// ---------------------------------------------------------------------------
void Kangaroo::CanonicalizeHost(Point* P, uint32_t* outE, uint32_t* outNeg) {

#ifdef DEMO_BOUNDED
    * outE = 0;
    *outNeg = 0;
    return;
#elif defined(NEGATION_ONLY)
    {
        *outE = 0;
        uint32_t n0 = 0;
        if (P->y.IsGreater(&PHALF_H)) { P->y.ModNeg(); n0 = 1; }
        *outNeg = n0;
        return;
    }
#else
    Int x1, x2;
    x1.Set(&P->x); x1.ModMulK1(&BETA_H);
    x2.Set(&P->x); x2.ModMulK1(&BETA2_H);

    uint32_t e = 0;
    if (x1.IsLower(&P->x)) e = 1;
    if (e == 0) { if (x2.IsLower(&P->x)) e = 2; }
    else { if (x2.IsLower(&x1))   e = 2; }

    if (e == 1)      P->x.Set(&x1);
    else if (e == 2) P->x.Set(&x2);

    uint32_t neg = 0;
    if (P->y.IsGreater(&PHALF_H)) { P->y.ModNeg(); neg = 1; }

    *outE = e;
    *outNeg = neg;
#endif
}

// ---------------------------------------------------------------------------
// Възстановяване на истинския скалар:  a_true = s * lambda^e * A  (mod n)
// ---------------------------------------------------------------------------
void Kangaroo::UnpackScalar(Int* A, uint32_t st, Int* aOut) {
    uint32_t e = ST_E(st);
    uint32_t neg = ST_NEG(st);

    aOut->Set(A);
    if (e == 1)      aOut->ModMulK1order(&LAMBDA);
    else if (e == 2) aOut->ModMulK1order(&LAMBDA2);
    if (neg)         aOut->ModNegK1order();
}


// ---------------------------------------------------------------------------
// k * P за произволна точка P (double-and-add).
// Secp256K1 предлага само k*G, затова тук е ръчно. Използва се само в
// SelfTest -- не е в горещия път.
// ---------------------------------------------------------------------------
static Point ScalarMultPoint(Secp256K1* secp, Int* k, Point Pin) {

    Point R;
    bool started = false;

    Point Q = Pin;
    Q.z.SetInt32(1);

    int bits = k->GetBitLength();
    for (int i = 0; i < bits; i++) {
        if (k->GetBit(i)) {
            if (!started) { R = Q; started = true; }
            else { R = secp->AddDirect(R, Q); }
        }
        Q = secp->DoubleDirect(Q);
    }

    if (!started) R.Clear();
    return R;
}

// ---------------------------------------------------------------------------
// Зареждане на таргетите
// ---------------------------------------------------------------------------
bool Kangaroo::ParseConfigFile(std::string& fileName) {

    keysToSearch.clear();
    foundTargets.clear();

#ifdef DEMO_BOUNDED
    {
        Int testPriv;
        testPriv.SetInt32(0xABCDE);
        Point testPub = secp->ComputePublicKey(&testPriv);
        keysToSearch.push_back(testPub);
        ::printf("!!! DEMO MODE (bounded 2^80) !!!\n");
        //::printf("Target generated from private key 0xABCDE\n");
        return true;
    }
#else
    std::ifstream file(fileName);
    if (!file.is_open()) {
        ::printf("Cannot open file: %s\n", fileName.c_str());
        return false;
    }

    std::string line;
    int count = 0, bad = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
            line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (line.empty()) continue;

        Point p;
        bool compressed;
        if (!secp->ParsePublicKeyHex(line, p, compressed)) { bad++; continue; }

        keysToSearch.push_back(p);
        count++;
        if (count >= 0x07FFFFFF) {          // ограничение на tIdx полето (27 бита)
            ::printf("Target limit reached (134M)\n");
            break;
        }
    }
    file.close();

    ::printf("Loaded %d target public keys", count);
    if (bad) ::printf(" (%d invalid lines skipped)", bad);
    ::printf("\n");
    return count > 0;
#endif
}

bool Kangaroo::IsDP(uint64_t x) { return (x & dMask) == 0; }

void Kangaroo::SetDP(int size) {
    dpSize = size;
    dMask = (dpSize == 0) ? 0 : ((1ULL << dpSize) - 1);
    ::printf("DP size: %d [Mask: 0x%016llX]\n", dpSize, (unsigned long long)dMask);
}

// ---------------------------------------------------------------------------
// WIF изход
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// WIF (компресиран) за произволен мрежов префикс
// ---------------------------------------------------------------------------
static std::string MakeWIF(Int* pk, unsigned char netPrefix) {

    static const char* B58 =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    unsigned char payload[34];
    payload[0] = netPrefix;
    pk->Get32Bytes(payload + 1);
    payload[33] = 0x01;                 // компресиран ключ

    unsigned char hash[32];
    sha256d(payload, 34, hash);

    unsigned char full[38];
    memcpy(full, payload, 34);
    memcpy(full + 34, hash, 4);

    int zeros = 0;
    while (zeros < 38 && full[zeros] == 0) zeros++;

    unsigned char buf[80] = { 0 };
    int bufLen = 0;
    for (int i = zeros; i < 38; i++) {
        int carry = full[i];
        for (int j = 0; j < bufLen; j++) {
            carry += 256 * buf[j];
            buf[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) { buf[bufLen++] = carry % 58; carry /= 58; }
    }

    char wif[80] = { 0 };
    int wifLen = 0;
    for (int i = 0; i < zeros; i++) wif[wifLen++] = '1';
    for (int i = bufLen - 1; i >= 0; i--) wif[wifLen++] = B58[buf[i]];
    wif[wifLen] = 0;

    return std::string(wif);
}
bool Kangaroo::Output(Int* pk, uint32_t tIdx) {

    struct NetInfo { const char* name; unsigned char prefix; };
    static const NetInfo nets[] = {
        { "BTC / BCH / ZEC", 0x80 },
        { "LTC",             0xB0 },
        { "DOGE",            0x9E },
        { "DASH",            0xCC }
    };
    const int nbNet = sizeof(nets) / sizeof(nets[0]);

    Point pub = secp->ComputePublicKey(pk);
    std::string pubC = secp->GetPublicKeyHex(true, pub);
    std::string pubU = secp->GetPublicKeyHex(false, pub);

    std::string outFileName;
    if (outputFile.length() > 0) {
        outFileName = outputFile;
    }
    else {
#ifdef WIN64
        char* userProfile = getenv("USERPROFILE");
        outFileName = userProfile ? std::string(userProfile) + "\\Desktop\\FOUND_KEYS.txt"
            : std::string("FOUND_KEYS.txt");
#else
        char* home = getenv("HOME");
        outFileName = home ? std::string(home) + "/Desktop/FOUND_KEYS.txt"
            : std::string("FOUND_KEYS.txt");
#endif
    }

    FILE* f = fopen(outFileName.c_str(), "a");
    bool needClose = (f != NULL);
    if (!f) f = stdout;

    ::fprintf(f, "=== Target #%u ===\n", tIdx);
    ::fprintf(f, "HEX        : 0x%s\n", pk->GetBase16().c_str());
    ::fprintf(f, "PubKey (c) : %s\n", pubC.c_str());
    ::fprintf(f, "PubKey (u) : %s\n", pubU.c_str());
    for (int i = 0; i < nbNet; i++)
        ::fprintf(f, "WIF %-16s: %s\n", nets[i].name,
            MakeWIF(pk, nets[i].prefix).c_str());
    ::fprintf(f, "\n");

    if (needClose) fclose(f);

    ::printf("\n>>> FOUND target #%u | HEX 0x%s\n",
        tIdx, pk->GetBase16().c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Решаване на колизия
//
// Всяка разходка е R = phi^e( +-( A*G + B*T ) ), B in {0,1}.
// След UnpackScalar имаме истинските a1, a2.
// Полезна е само tame(B=0) x wild(B=1):   T = (a_tame - a_wild) * G
// ---------------------------------------------------------------------------
bool Kangaroo::SolveCollision(COLLISION& c) {

    Int A1, A2, a1, a2;
    A1.SetInt32(0); memcpy(A1.bits64, c.a1, 32);
    A2.SetInt32(0); memcpy(A2.bits64, c.a2, 32);

    UnpackScalar(&A1, c.st1, &a1);
    UnpackScalar(&A2, c.st2, &a2);

    uint32_t b1 = ST_B(c.st1), b2 = ST_B(c.st2);
    uint32_t t1 = ST_TIDX(c.st1), t2 = ST_TIDX(c.st2);

    Int key;
    uint32_t tIdx;
    uint32_t wildSt;

    if (b1 == 0 && b2 == 1) {
        key.Set(&a1); key.ModSubK1order(&a2);   // a_tame - a_wild
        tIdx = t2;
        wildSt = c.st2;
    }
    else if (b2 == 0 && b1 == 1) {
        key.Set(&a2); key.ModSubK1order(&a1);
        tIdx = t1;
        wildSt = c.st1;
    }
    else {
        // tame x tame, или wild x wild -- не дава решение
        return false;
    }

    // ------------------------------------------------------------------
    // КОРЕКЦИЯ ЗА GLV.
    //
    // Истинският коефициент пред таргета при wild разходката НЕ е 1, а
    //     b_true = sigma * lambda^e
    // Уравнението е  a_tame*G = a_wild*G + b_true*T,  откъдето
    //     k = (a_tame - a_wild) * b_true^-1
    //
    // Обратното се знае явно: lambda^-1 == lambda^2, lambda^-2 == lambda,
    // а sigma^-1 == sigma. Не е нужен модулен инверс.
    // ------------------------------------------------------------------
    uint32_t eW = ST_E(wildSt);
    uint32_t negW = ST_NEG(wildSt);

    Int invB;
    if (eW == 1)      invB.Set(&LAMBDA2);   // lambda^-1
    else if (eW == 2) invB.Set(&LAMBDA);    // lambda^-2
    else              invB.SetInt32(1);
    if (negW) invB.ModNegK1order();

    key.ModMulK1order(&invB);

    if (tIdx >= keysToSearch.size()) return false;
    if (foundTargets.count(tIdx)) return false;

    Point P = secp->ComputePublicKey(&key);
    if (!P.x.IsEqual(&keysToSearch[tIdx].x)) {
        Int kneg; kneg.Set(&key); kneg.ModNegK1order();
        Point Pn = secp->ComputePublicKey(&kneg);
        if (!Pn.x.IsEqual(&keysToSearch[tIdx].x)) return false;
        key.Set(&kneg);
    }

    foundTargets.insert(tIdx);
    Output(&key, tIdx);

    ::printf("    (%zu / %zu targets solved)\n",
        foundTargets.size(), keysToSearch.size());

    if (foundTargets.size() >= keysToSearch.size()) endOfSearch = true;
    return true;
}

// ---------------------------------------------------------------------------
// SELF-TEST -- проверява инварианта на CPU, без GPU.
//
// Инвариант: R == (s * lambda^e * A) * G  +  (s * lambda^e * B) * T
//
// Това е ЕДИНСТВЕНИЯТ надежден тест за GLV счетоводството в пълен диапазон.
// End-to-end тест "намери ключа" е невъзможен в Ро режим.
// ---------------------------------------------------------------------------
bool Kangaroo::SelfTest() {

    ::printf("\n=== SELF-TEST ===\n");

    Int kSecret;
    kSecret.SetInt32(0xABCDE);
    Point T = secp->ComputePublicKey(&kSecret);

    keysToSearch.clear();
    keysToSearch.push_back(T);
    foundTargets.clear();

    InitRange();
    CreateJumpTable();

    bool ok = true;

    // ==================================================================
    // ТЕСТ 1 -- инвариант на разходката
    //   R  ==  (s * lambda^e * A) * G  +  (s * lambda^e * B) * T
    // ==================================================================
    ::printf("--- 1. walk invariant ---\n");
    {
        Int A; A.Rand(255);
        Int B; B.SetInt32(1);              // wild разходка

        Point P = secp->ComputePublicKey(&A);
        P = secp->AddDirect(P, T);

        uint32_t e, neg;
        CanonicalizeHost(&P, &e, &neg);

        const int NSTEP = 32;
        for (int step = 0; step < NSTEP; step++) {

            uint32_t st = ST_PACK(e, neg, 1, 0);

            Int aTrue, bTrue;
            UnpackScalar(&A, st, &aTrue);
            UnpackScalar(&B, st, &bTrue);

            Point lhs = secp->ComputePublicKey(&aTrue);
            if (!bTrue.IsZero()) {
                Point bT = ScalarMultPoint(secp, &bTrue, T);
                lhs = secp->AddDirect(lhs, bT);
            }

            if (!lhs.x.IsEqual(&P.x)) {
                ::printf("  [FAIL] step %d (e=%u neg=%u)\n", step, e, neg);
                ::printf("     walk  x = %s\n", P.x.GetBase16().c_str());
                ::printf("     calc  x = %s\n", lhs.x.GetBase16().c_str());
                ok = false;
                break;
            }

            // --- една стъпка, точно както в ядрото ---
            uint32_t jmp = (uint32_t)(P.x.bits64[0]) & (NB_JUMP - 1);

            Point J;
            J.x.Set(&jumpPointx[jmp]);
            J.y.Set(&jumpPointy[jmp]);
            J.z.SetInt32(1);
            P = secp->AddDirect(P, J);

            // A += sigma * lambda^-e * c_i   (знакът е ПРЕДИ канонизацията)
            if (neg == 0) A.ModAddK1order(&jCGtable[e * NB_JUMP + jmp]);
            else          A.ModSubK1order(&jCGtable[e * NB_JUMP + jmp]);

            uint32_t dE, dNeg;
            CanonicalizeHost(&P, &dE, &dNeg);
            e = (e + dE) % 3;
            neg = neg ^ dNeg;
        }

        if (ok) ::printf("  [PASS] invariant held %d steps\n", NSTEP);
    }

    // ==================================================================
    // ТЕСТ 2 -- решаване на колизия, за всичките 6 класа на wild страната
    //
    // Строим ИСТИНСКА колизия: tame и wild в една и съща точка.
    //   tame:  R = a_tame * G                       (B = 0)
    //   wild:  R = a_wild * G + b_wild * T          (B = 1)
    //   =>     a_wild = a_tame - b_wild * k
    // където b_wild = sigma * lambda^e -- истинският коефициент, а
    // съхраненият скалар е  A = sigma * lambda^-e * a_true.
    // ==================================================================
    ::printf("--- 2. collision solve (all 6 GLV classes) ---\n");

    for (uint32_t eW = 0; eW < 3; eW++) {
        for (uint32_t negW = 0; negW < 2; negW++) {

            foundTargets.clear();

            // b_wild = sigma * lambda^eW
            Int bWild;
            if (eW == 1)      bWild.Set(&LAMBDA);
            else if (eW == 2) bWild.Set(&LAMBDA2);
            else              bWild.SetInt32(1);
            if (negW) bWild.ModNegK1order();

            Int aTame; aTame.Rand(200);

            // a_wild = a_tame - b_wild * k
            Int bk; bk.Set(&bWild); bk.ModMulK1order(&kSecret);
            Int aWild; aWild.Set(&aTame); aWild.ModSubK1order(&bk);

            // съхранените скалари: A = sigma * lambda^-e * a_true
            Int invL;
            if (eW == 1)      invL.Set(&LAMBDA2);
            else if (eW == 2) invL.Set(&LAMBDA);
            else              invL.SetInt32(1);
            if (negW) invL.ModNegK1order();

            Int Awild; Awild.Set(&aWild); Awild.ModMulK1order(&invL);
            Int Atame; Atame.Set(&aTame);          // tame: e=0, neg=0

            COLLISION c;
            memset(&c, 0, sizeof(c));
            memcpy(c.a1, Atame.bits64, 32);
            c.st1 = ST_PACK(0, 0, 0, 0);              // tame
            memcpy(c.a2, Awild.bits64, 32);
            c.st2 = ST_PACK(eW, negW, 1, 0);          // wild

            bool solved = SolveCollision(c);
            ::printf("  [%s] e=%u neg=%u\n", solved ? "PASS" : "FAIL", eW, negW);
            if (!solved) ok = false;
        }
    }

    ::printf("=== SELF-TEST %s ===\n\n", ok ? "PASSED" : "FAILED");
    return ok;
}

// ---------------------------------------------------------------------------

void Kangaroo::SolveKeyCPU(TH_PARAM* ph) { ph->isRunning = false; }

// ---------------------------------------------------------------------------

namespace {
    struct HostDP {
        uint64_t x0, x1;
        uint64_t a0, a1, a2, a3;
        uint32_t st;
        uint32_t used;             // 0 = празен слот (calloc нулира)
    };
    static HostDP* gTab = nullptr;
    static uint64_t gTabMask = 0;         // размер-1 (размерът е степен на 2)
    static uint64_t gTabSize = 0;
    static uint64_t gTabCount = 0;         // брой заети слота
    static bool     gTabFullWarned = false;
    static const int POOL_PROBE = 64;      // макс. линейно пробване

    static inline uint64_t PoolHash(uint64_t x0, uint64_t x1) {
        uint64_t h = x0 ^ (x1 * 0x9E3779B97F4A7C15ULL);
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27; h *= 0x94D049BB133111EBULL; h ^= h >> 31;
        return h;
    }

    // --- постави запис в таблицата (без решаване), ползва се при зареждане от диск ---
    static void PoolPlace(uint64_t x0, uint64_t x1, uint64_t a0, uint64_t a1,
        uint64_t a2, uint64_t a3, uint32_t st) {
        uint64_t slot = PoolHash(x0, x1) & gTabMask;
        for (int probe = 0; probe < POOL_PROBE; probe++) {
            HostDP& e = gTab[slot];
            if (!e.used) {
                e.x0 = x0; e.x1 = x1; e.a0 = a0; e.a1 = a1; e.a2 = a2; e.a3 = a3; e.st = st;
                e.used = 1; gTabCount++; return;
            }
            if (e.x0 == x0 && e.x1 == x1) return;   // вече е там
            slot = (slot + 1) & gTabMask;
        }
    }

    // файл формат: magic "RPOOL1", u32 POOL_BITS, u64 count, после count записа
    //   (u64 x0, x1, a0, a1, a2, a3, u32 st) = 52 B/запис
#define POOL_MAGIC 0x3150 /* "P1" */
    static void PoolSave(const char* fn) {
        if (!gTab) return;
        FILE* f = fopen(fn, "wb");
        if (!f) { ::printf("PoolSave: ne moga da otvorq %s\n", fn); return; }
        uint32_t bits = POOL_BITS, magic = POOL_MAGIC;
        fwrite(&magic, 4, 1, f); fwrite(&bits, 4, 1, f); fwrite(&gTabCount, 8, 1, f);
        uint64_t written = 0;
        for (uint64_t s = 0; s < gTabSize; s++) {
            HostDP& e = gTab[s];
            if (!e.used) continue;
            fwrite(&e.x0, 8, 1, f); fwrite(&e.x1, 8, 1, f);
            fwrite(&e.a0, 8, 1, f); fwrite(&e.a1, 8, 1, f);
            fwrite(&e.a2, 8, 1, f); fwrite(&e.a3, 8, 1, f);
            fwrite(&e.st, 4, 1, f); written++;
        }
        fclose(f);
        ::printf("[pool] zapisani %llu DP v %s\n", (unsigned long long)written, fn);
    }
    static void PoolLoad(const char* fn) {
        FILE* f = fopen(fn, "rb");
        if (!f) return;                                   // nqma fail -> prust start
        uint32_t bits = 0, magic = 0; uint64_t count = 0;
        if (fread(&magic, 4, 1, f) != 1 || magic != POOL_MAGIC) { fclose(f); return; }
        fread(&bits, 4, 1, f); fread(&count, 8, 1, f);
        uint64_t loaded = 0;
        for (uint64_t i = 0; i < count; i++) {
            uint64_t x0, x1, a0, a1, a2, a3; uint32_t st;
            if (fread(&x0, 8, 1, f) != 1) break;
            fread(&x1, 8, 1, f); fread(&a0, 8, 1, f); fread(&a1, 8, 1, f);
            fread(&a2, 8, 1, f); fread(&a3, 8, 1, f); fread(&st, 4, 1, f);
            PoolPlace(x0, x1, a0, a1, a2, a3, st); loaded++;
        }
        fclose(f);
        ::printf("[pool] zaredeni %llu DP ot %s (prodaljavame ottam)\n",
            (unsigned long long)loaded, fn);
    }

    static void PoolInit() {
        if (gTab) return;
        gTabSize = 1ULL << POOL_BITS;
        gTabMask = gTabSize - 1;
        gTab = (HostDP*)calloc(gTabSize, sizeof(HostDP));   // нулиран -> всички used=0
        if (!gTab) {
            ::printf("PoolInit: calloc failed (%llu MB)\n",
                (unsigned long long)(gTabSize * sizeof(HostDP) / 1048576ULL)); return;
        }
        PoolLoad("dp_pool.bin");                            // продължи от предния пуск, ако има
    }
}

// ----------------------------------------------------------------------------

void Kangaroo::SolveKeyGPU(TH_PARAM* ph) {

#ifdef WITHGPU
    std::vector<GPUEngine::GpuDP> dps;

    // poolMode=true в конструктора -> GHT не се заделя (спестява ~GBs VRAM)
    GPUEngine* gpu = new GPUEngine(ph->gridSizeX, ph->gridSizeY, ph->gpuId, 65536, true);

    // една обща таблица за всички GPU нишки (пази се с ghMutex)
    { LOCK(ghMutex); PoolInit(); UNLOCK(ghMutex); }

    ::printf("GPU: %s (%d MB used)\n", gpu->deviceName.c_str(), gpu->GetMemory());
    double t0 = Timer::get_tick();

    if (ph->px == NULL) {
        uint64_t nbThread = gpu->GetNbThread();
        ph->px = new Int[ph->nbKangaroo];
        ph->py = new Int[ph->nbKangaroo];
        ph->a = new Int[ph->nbKangaroo];
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
    ::printf("SolveKeyGPUc Thread GPU#%d: 2^%.2f walks [%.1fs]\n",
        ph->gpuId, log2((double)ph->nbKangaroo), (t1 - t0));
    ph->hasStarted = true;

    double lastFlush = Timer::get_tick();
    const double FLUSH_SEC = 300.0;              // запис на диск на всеки 5 мин

    while (!endOfSearch) {

        if (!gpu->LaunchPool(dps)) break;        // пуска ядрото + връща DP от предния пуск
        counters[0x80L + ph->threadId] += ph->nbKangaroo * NB_RUN;

        if (!dps.empty()) {
            LOCK(ghMutex);
            for (size_t i = 0; i < dps.size() && !endOfSearch; i++) {
                const GPUEngine::GpuDP& d = dps[i];
                uint64_t slot = PoolHash(d.x[0], d.x[1]) & gTabMask;
                for (int probe = 0; probe < POOL_PROBE; probe++) {
                    HostDP& e = gTab[slot];
                    if (!e.used) {
                        // празен -> вмъкни (ако таблицата не е почти пълна)
                        if (gTabCount * 10 < gTabSize * 7) {   // под 70%
                            e.x0 = d.x[0]; e.x1 = d.x[1];
                            e.a0 = d.a[0]; e.a1 = d.a[1]; e.a2 = d.a[2]; e.a3 = d.a[3];
                            e.st = d.st; e.used = 1; gTabCount++;
                        }
                        else if (!gTabFullWarned) {
                            ::printf("\nPool tablica >70%% pulna (2^%d) -> spirame vmakvane; "
                                "uvelichi POOL_BITS ili -d\n", POOL_BITS);
                            gTabFullWarned = true;
                        }
                        break;
                    }
                    if (e.x0 == d.x[0] && e.x1 == d.x[1]) {
                        bool same = (e.a0 == d.a[0] && e.a1 == d.a[1] &&
                            e.a2 == d.a[2] && e.a3 == d.a[3] && e.st == d.st);
                        if (!same) {
                            // РЕАЛНА колизия -> решаваме ключа
                            COLLISION c;
                            c.x[0] = d.x[0]; c.x[1] = d.x[1];
                            c.a1[0] = d.a[0]; c.a1[1] = d.a[1]; c.a1[2] = d.a[2]; c.a1[3] = d.a[3]; c.st1 = d.st;
                            c.a2[0] = e.a0;  c.a2[1] = e.a1;  c.a2[2] = e.a2;  c.a2[3] = e.a3;  c.st2 = e.st;
                            if (!SolveCollision(c)) collisionInSameHerd++;
                        }
                        break;   // дубликат или вече обработена колизия
                    }
                    slot = (slot + 1) & gTabMask;   // линейно пробване
                }
            }
            UNLOCK(ghMutex);
        }

        // --- рестарт на заклещените разходки (per-карта) ---
        // ============================================================================
// WATCHDOG RESET — пач за Kangaroo::SolveKeyGPU (Ро, пълен диапазон)
// ----------------------------------------------------------------------------
// ПРОБЛЕМ: в пълен 2^256 диапазон разходките не влизат в 2-цикли, затова
// хардуерният reset (cycTot==32) почти никога не щрака. Разходките вечно вървят
// по същите детерминистични вериги -> раждат краен набор DP -> броячът забива.
//
// РЕШЕНИЕ: периодично рестартирай малка част от разходките с НОВИ случайни
// начални точки. Така постоянно вливаш свежи DP вериги -> броячът расте без
// застой, DP са различни, покриват различни части на диапазона.
//
// КАК СЕ ПОЛЗВА: замести целия блок
//   "// --- рестарт на заклещените разходки (per-карта) ---" { ... }
// в SolveKeyGPU със следния код.
//
// ВАЖНО (честно): това прави DP да растат стабилно и да са различни. НЕ
// увеличава шанса за колизия в пълен диапазон — той остава ~2^-117. По-чест
// reset = повече различни DP, но по-къси разходки; затова държим процента МАЛЪК.
// ============================================================================

        // --- 1) хардуерен reset на заклещените (оставаме го както е) ---
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

        // --- 2) WATCHDOG: подновявай малка част от разходките всеки launch ---
        //     Това държи DP да растат непрекъснато в пълен диапазон.
        {
            const uint64_t totalWalks =
                (uint64_t)gpu->GetNbThread() * gpu->GetGroupSize();

            // колко да подновим на launch:
            //   RESET_FRAC = колко части от разходките (1/RESET_FRAC) на launch.
            //   200 -> 0.5% на launch. На по-голямо число = по-рядко подновяване.
            //   НЕ го прави твърде малко (по-агресивният reset чупи дълги разходки).
            const uint64_t RESET_FRAC = 200;                 // 0.5% на launch
            uint64_t nRenew = totalWalks / RESET_FRAC;
            if (nRenew == 0) nRenew = 1;
            // таван, за да не надхвърли RESET_MAX буфера (оставяме запас)
            if (nRenew > 4096) nRenew = 4096;

            // въртящ курсор, за да обходим ВСИЧКИ разходки с времето (не само едни)
            static thread_local uint64_t renewCursor = 0;

            for (uint64_t k = 0; k < nRenew; k++) {
                uint32_t idx = (uint32_t)(renewCursor % totalWalks);
                renewCursor++;

                Int n_px, n_py, n_a; uint32_t n_st;
                // редуваме tame/wild по четност на индекса (както при CreateHerd)
                CreateHerd(1, &n_px, &n_py, &n_a, &n_st, (int)(idx & 1), false);
                gpu->SetKangaroo(idx, &n_px, &n_py, &n_a, n_st);
            }
        }

        // --- периодичен запис на DP базата (само нишка 0) ---
        if (ph->threadId == 0) {
            double now = Timer::get_tick();
            if (now - lastFlush >= FLUSH_SEC) {
                LOCK(ghMutex);
                PoolSave("dp_pool.bin");          // запис под lock (кратък престой на картите)
                UNLOCK(ghMutex);
                lastFlush = now;
            }
        }
    }

    // --- финален запис при чисто спиране / намерен ключ ---
    if (ph->threadId == 0) {
        LOCK(ghMutex);
        PoolSave("dp_pool.bin");
        UNLOCK(ghMutex);
    }

    delete gpu;
#else
    ph->hasStarted = true;
#endif
    ph->isRunning = false;
}



#ifdef WIN64
DWORD WINAPI _SolveKeyCPU(LPVOID lpParam) {
#else
void* _SolveKeyCPU(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->obj->SolveKeyCPU(p);
    return 0;
}

#ifdef WIN64
DWORD WINAPI _SolveKeyGPU(LPVOID lpParam) {
#else
void* _SolveKeyGPU(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->obj->SolveKeyGPU(p);
    return 0;
}

// ---------------------------------------------------------------------------
// Стартови точки
// ---------------------------------------------------------------------------
void Kangaroo::CreateHerd(int nbKangaroo, Int * px, Int * py, Int * a,
    uint32_t * state, int firstType, bool lock) {

    if (lock) LOCK(ghMutex);

    size_t nbTarget = keysToSearch.size();
    if (nbTarget == 0) nbTarget = 1;

#ifdef DEMO_BOUNDED
    Int halfW;
    halfW.SetInt32(0);
    halfW.bits64[0] = (1ULL << (rangePower - 1));
#endif

    for (int64_t j = 0; j < (int64_t)nbKangaroo; j++) {

        int type = (firstType + (int)j) & 1;
        uint32_t tIdx = 0;
        Point P;

#ifdef DEMO_BOUNDED
        Int rnd;
        rnd.Rand(rangePower - 1);
        if (type == TAME) {
            a[j].Set(&halfW);
            a[j].Add(&rnd);
            P = secp->ComputePublicKey(&a[j]);
        }
        else {
            a[j].Set(&rnd);
            P = secp->ComputePublicKey(&a[j]);
            P = secp->AddDirect(P, keysToSearch[0]);
        }
#else
        a[j].Rand(255);
        if (a[j].IsZero()) a[j].SetInt32(1);
        P = secp->ComputePublicKey(&a[j]);
        if (type == WILD) {
            tIdx = (uint32_t)(((uint64_t)rand() << 31 ^ (uint64_t)rand()) % nbTarget);
            P = secp->AddDirect(P, keysToSearch[tIdx]);
        }
#endif

        uint32_t e = 0, neg = 0;
        CanonicalizeHost(&P, &e, &neg);

        px[j].Set(&P.x);
        py[j].Set(&P.y);
        state[j] = ST_PACK(e, neg, type, tIdx);
    }

    if (lock) UNLOCK(ghMutex);
}

// ---------------------------------------------------------------------------
// Таблица скокове + мащабирани скалари jCG[e][i] = lambda^-e * c_i mod n
// ---------------------------------------------------------------------------
void Kangaroo::CreateJumpTable() {

#ifdef DEMO_BOUNDED
    int jumpBits = rangePower / 2;      // оптимален скок ~sqrt(range)
    if (jumpBits < 2) jumpBits = 2;     // долна граница за много малки диапазони
#else
    int jumpBits = 128;                 // пълен 2^256
#endif


    for (int i = 0; i < NB_JUMP; i++) {

        jumpA[i].Rand(jumpBits);
        if (jumpA[i].IsZero()) jumpA[i].SetInt32(1);
        jumpB[i].SetInt32(0);

        Point Pa = secp->ComputePublicKey(&jumpA[i]);
        jumpPointx[i].Set(&Pa.x);
        jumpPointy[i].Set(&Pa.y);

        // e = 0 : c_i
        jCGtable[0 * NB_JUMP + i].Set(&jumpA[i]);
        // e = 1 : lambda^-1 * c_i  ==  lambda^2 * c_i
        jCGtable[1 * NB_JUMP + i].Set(&jumpA[i]);
        jCGtable[1 * NB_JUMP + i].ModMulK1order(&LAMBDA2);
        // e = 2 : lambda^-2 * c_i  ==  lambda * c_i
        jCGtable[2 * NB_JUMP + i].Set(&jumpA[i]);
        jCGtable[2 * NB_JUMP + i].ModMulK1order(&LAMBDA);
    }
}

// ---------------------------------------------------------------------------

// ============================================================================
// ПАЧ: ComputeExpected — отчита броя таргети (мултитаргет предимство)
// ----------------------------------------------------------------------------
// Старата формула: op = sqrt(pi/2 * 2^256) = 2^128.33 ВИНАГИ, игнорира таргетите.
// При K таргета очакваните операции падат ~sqrt(K) пъти:
//     op = sqrt(pi/2 * N / K)
// Пример: K = 4.6M (~2^22.1) -> op пада от 2^128.33 на ~2^117.3.
//
// Замести цялата функция Kangaroo::ComputeExpected с тази.
// ============================================================================

void Kangaroo::ComputeExpected(double dp, double* op, double* ram, double* overHead) {
#ifdef DEMO_BOUNDED
    double N = pow(2.0, (double)rangePower);
#else
    double N = pow(2.0, 256.0);
#endif

    // --- мултитаргет: K независими таргета намаляват очакваните операции ~sqrt(K) ---
    double K = (double)keysToSearch.size();
    if (K < 1.0) K = 1.0;
    N = N / K;                           // ефективно пространство на таргет

    *op = sqrt(N * M_PI / 2.0);

    if (ram) *ram = 0.0;
    if (overHead) *overHead = 0.0;
}

void Kangaroo::InitRange() {
#ifdef DEMO_BOUNDED
    rangePower = 80;
#else
    rangePower = 256;
#endif
}

// ---------------------------------------------------------------------------

void Kangaroo::Run(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize) {

#ifdef NEGATION_ONLY
    ::printf("[BUILD] NEGATION_ONLY = ON\n");
#else
    ::printf("[BUILD] GLV active\n");
#endif

    double t0 = Timer::get_tick();
    startTime = t0;

    nbCPUThread = nbThread;
    nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
    totalRW = 0;

    uint64_t totalThread = (uint64_t)nbCPUThread + (uint64_t)nbGPUThread;
    if (totalThread == 0) {
        ::printf("No CPU or GPU thread, exiting.\n");
        ::exit(0);
    }

    TH_PARAM* params = (TH_PARAM*)malloc(totalThread * sizeof(TH_PARAM));
    THREAD_HANDLE* thHandles = (THREAD_HANDLE*)malloc(totalThread * sizeof(THREAD_HANDLE));
    if (!params || !thHandles) { ::printf("Memory allocation failed\n"); return; }

    memset(params, 0, totalThread * sizeof(TH_PARAM));
    memset(counters, 0, sizeof(counters));

#ifdef WITHGPU
    for (int i = 0; i < nbGPUThread; i++) {
        int x = gridSize[2ULL * i];
        int y = gridSize[2ULL * i + 1ULL];
        if (GPUEngine::GetGridSize(gpuId[i], &x, &y)) {
            params[nbCPUThread + i].gridSizeX = x;
            params[nbCPUThread + i].gridSizeY = y;
        }
        params[nbCPUThread + i].nbKangaroo = (uint64_t)GPU_GRP_SIZE * x * y;
        totalRW += params[nbCPUThread + i].nbKangaroo;
    }
#endif
    totalRW += nbCPUThread * (uint64_t)CPU_GRP_SIZE;

    InitRange();
    CreateJumpTable();

#ifdef DEMO_BOUNDED
    ::printf("Mode: BOUNDED KANGAROO (demo, range 2^%d)\n", rangePower);
#elif defined(NEGATION_ONLY)
    ::printf("Mode: POLLARD RHO, negation map only\n");
#else
    ::printf("Mode: POLLARD RHO, GLV + negation\n");
#endif

    ::printf("Number of walks: 2^%.2f\n", log2((double)totalRW));
    double eOp, eMem;
    ComputeExpected((double)initDPSize, &eOp, &eMem, NULL);
    ::printf("Expected operations: 2^%.2f\n", log2(eOp));
    //::printf("Expected to find - REALLY SOON!!!\n");
    ::printf("EASY targets: %zu\n", keysToSearch.size());
    //::printf("Targets - EASY TO FIND!\n");

    if (initDPSize < 0) initDPSize = 22;
    SetDP(initDPSize);

    endOfSearch = false;
    collisionInSameHerd = 0;

    for (int i = 0; i < nbCPUThread; i++) {
        params[i].threadId = i;
        params[i].isRunning = false;
        params[i].hasStarted = true;
    }

#ifdef WITHGPU
    for (int i = 0; i < nbGPUThread; i++) {
        int id = nbCPUThread + i;
        params[id].threadId = i;
        params[id].isRunning = true;
        params[id].gpuId = gpuId[i];
        thHandles[id] = LaunchThread(_SolveKeyGPU, params + id);
    }
#endif

    Process(params, "MK/s");

    JoinThreads(thHandles + nbCPUThread, nbGPUThread);
    FreeHandles(thHandles + nbCPUThread, nbGPUThread);

    free(params);
    free(thHandles);

    double t1 = Timer::get_tick();
    ::printf("\nDone: Total time %s\n", GetTimeStr(t1 - t0 + offsetTime).c_str());
}

// ---------------------------------------------------------------------------
// Оставено за съвместимост с ProcessServer в Thread.cpp
// ---------------------------------------------------------------------------
bool Kangaroo::AddToTable(uint64_t h, int128_t * x, int128_t * d) {
    return true;
}