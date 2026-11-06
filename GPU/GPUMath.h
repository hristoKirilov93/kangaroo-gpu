/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
* Fully Restored & Modified for Pollard's Rho implementation.
*/

#ifndef GPUMATHH
#define GPUMATHH

#include <stdint.h>
#include <cuda_runtime.h>
#include "../Constants.h"

// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDS(r,a,b,c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

__device__ __constant__ uint64_t jPx[NB_JUMP][4];
__device__ __constant__ uint64_t jPy[NB_JUMP][4];
__device__ __constant__ uint64_t jA[NB_JUMP][4];
__device__ __constant__ uint64_t jB[NB_JUMP][4];

__device__ __constant__ uint64_t _O[] = { 0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL };

#define HSIZE (GRP_SIZE / 2 - 1)
#define MM64 0xD838091DD2253531ULL

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x
#define __sleft128(a,b,n) ((b)<<(n))|((a)>>(64-(n)))

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

#define Neg(r) {\
USUBO(r[0],0ULL,r[0]); \
USUBC(r[1],0ULL,r[1]); \
USUBC(r[2],0ULL,r[2]); \
USUBC(r[3],0ULL,r[3]); \
USUB(r[4],0ULL,r[4]); }

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}

__device__ void ModSub256(uint64_t* r, uint64_t* a, uint64_t* b) {
    uint64_t t;
    uint64_t T[4];
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);
}

__device__ void ModSub256(uint64_t* r, uint64_t* b) {
    uint64_t t;
    uint64_t T[4];
    USUBO(r[0], r[0], b[0]);
    USUBC(r[1], r[1], b[1]);
    USUBC(r[2], r[2], b[2]);
    USUBC(r[3], r[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);
}

__device__ __forceinline__ uint32_t ctz(uint64_t x) {
    uint32_t n;
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " brev.b64 tmp, %1;\n\t"
        " clz.b64 %0, tmp;\n\t"
        "}"
        : "=r"(n) : "l"(x));
    return n;
}

#define SWAP(tmp,x,y) tmp = x; x = y; y = tmp;
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void ShiftR62(uint64_t r[5]) {
    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    r[4] = (int64_t)(r[4]) >> 62;
}

__device__ void ShiftR62(uint64_t dest[5], uint64_t r[5], uint64_t carry) {
    dest[0] = (r[1] << 2) | (r[0] >> 62);
    dest[1] = (r[2] << 2) | (r[1] >> 62);
    dest[2] = (r[3] << 2) | (r[2] >> 62);
    dest[3] = (r[4] << 2) | (r[3] >> 62);
    dest[4] = (carry << 2) | (r[4] >> 62);
}

__device__ void _DivStep62(uint64_t u[5], uint64_t v[5], int32_t* pos, int64_t* uu, int64_t* uv, int64_t* vu, int64_t* vv) {
    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;

    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];

    uint64_t uh, vh;
    int64_t w, x, y, z;

    while (*pos > 0 && (u[*pos] | v[*pos]) == 0) (*pos)--;
    if (*pos == 0) {
        uh = u[0];
        vh = v[0];
    }
    else {
        uint32_t s = __clzll(u[*pos] | v[*pos]);
        if (s == 0) {
            uh = u[*pos];
            vh = v[*pos];
        }
        else {
            uh = __sleft128(u[*pos - 1], u[*pos], s);
            vh = __sleft128(v[*pos - 1], v[*pos], s);
        }
    }

    while (true) {
        zeros = ctz(v0 | (1ULL << bitCount));
        v0 >>= zeros;
        vh >>= zeros;
        *uu <<= zeros;
        *uv <<= zeros;
        bitCount -= zeros;

        if (bitCount == 0)
            break;

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(y, *uu, *vu);
            SWAP(z, *uv, *vv);
        }

        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;
    }
}

__device__ void IMult(uint64_t* r, uint64_t* a, int64_t b) {
    uint64_t t[NBBLOCK];
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    }
    else {
        Load(t, a);
    }
    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADD(r[4], t[3], b, r[4]);
}

__device__ uint64_t IMultC(uint64_t* r, uint64_t* a, int64_t b) {
    uint64_t t[NBBLOCK];
    uint64_t carry;
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    }
    else {
        Load(t, a);
    }
    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADDC(r[4], t[3], b, r[4]);
    MADDS(carry, t[4], b, 0ULL);
    return carry;
}

__device__ void MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, uint64_t* carry) {
    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t c1, c2;

    c1 = IMultC(t1, u, _11);
    c2 = IMultC(t2, v, _12);

    UADDO(dest[0], t1[0], t2[0]);
    UADDC(dest[1], t1[1], t2[1]);
    UADDC(dest[2], t1[2], t2[2]);
    UADDC(dest[3], t1[3], t2[3]);
    UADDC(dest[4], t1[4], t2[4]);
    UADD(*carry, c1, c2);
}

__device__ void MatrixVecMul(uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t t3[NBBLOCK];
    uint64_t t4[NBBLOCK];

    IMult(t1, u, _11);
    IMult(t2, v, _12);
    IMult(t3, u, _21);
    IMult(t4, v, _22);

    UADDO(u[0], t1[0], t2[0]);
    UADDC(u[1], t1[1], t2[1]);
    UADDC(u[2], t1[2], t2[2]);
    UADDC(u[3], t1[3], t2[3]);
    UADD(u[4], t1[4], t2[4]);

    UADDO(v[0], t3[0], t4[0]);
    UADDC(v[1], t3[1], t4[1]);
    UADDC(v[2], t3[2], t4[2]);
    UADDC(v[3], t3[3], t4[3]);
    UADD(v[4], t3[4], t4[4]);
}

__device__ uint64_t AddCh(uint64_t r[5], uint64_t a[5], uint64_t carry) {
    uint64_t carryOut;
    UADDO1(r[0], a[0]);
    UADDC1(r[1], a[1]);
    UADDC1(r[2], a[2]);
    UADDC1(r[3], a[3]);
    UADDC1(r[4], a[4]);
    UADD(carryOut, carry, 0ULL);
    return carryOut;
}

__device__ void MulP(uint64_t* r, uint64_t a) {
    uint64_t ah;
    uint64_t al;
    UMULLO(al, a, 0x1000003D1ULL);
    UMULHI(ah, a, 0x1000003D1ULL);
    USUBO(r[0], 0ULL, al);
    USUBC(r[1], 0ULL, ah);
    USUBC(r[2], 0ULL, 0ULL);
    USUBC(r[3], 0ULL, 0ULL);
    USUB(r[4], a, 0ULL);
}

__device__ __noinline__ void _ModInv(uint64_t* R) {
    int64_t  uu, uv, vu, vv;
    uint64_t mr0, ms0;
    int32_t  pos = NBBLOCK - 1;

    uint64_t u[NBBLOCK];
    uint64_t v[NBBLOCK];
    uint64_t r[NBBLOCK];
    uint64_t s[NBBLOCK];
    uint64_t tr[NBBLOCK];
    uint64_t ts[NBBLOCK];
    uint64_t r0[NBBLOCK];
    uint64_t s0[NBBLOCK];
    uint64_t carryR;
    uint64_t carryS;

    u[0] = 0xFFFFFFFEFFFFFC2F;
    u[1] = 0xFFFFFFFFFFFFFFFF;
    u[2] = 0xFFFFFFFFFFFFFFFF;
    u[3] = 0xFFFFFFFFFFFFFFFF;
    u[4] = 0;
    Load(v, R);
    r[0] = 0; s[0] = 1;
    r[1] = 0; s[1] = 0;
    r[2] = 0; s[2] = 0;
    r[3] = 0; s[3] = 0;
    r[4] = 0; s[4] = 0;

    while (true) {
        _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);
        MatrixVecMul(u, v, uu, uv, vu, vv);

        if (_IsNegative(u)) {
            Neg(u);
            uu = -uu;
            uv = -uv;
        }
        if (_IsNegative(v)) {
            Neg(v);
            vu = -vu;
            vv = -vv;
        }

        ShiftR62(u);
        ShiftR62(v);

        MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
        mr0 = (tr[0] * MM64) & MSK62;
        MulP(r0, mr0);
        carryR = AddCh(tr, r0, carryR);

        if (_IsZero(v)) {
            ShiftR62(r, tr, carryR);
            break;
        }
        else {
            MatrixVecMulHalf(ts, r, s, vu, vv, &carryS);
            ms0 = (ts[0] * MM64) & MSK62;
            MulP(s0, ms0);
            carryS = AddCh(ts, s0, carryS);
        }

        ShiftR62(r, tr, carryR);
        ShiftR62(s, ts, carryS);
    }

    if (!_IsOne(u)) {
        R[0] = 0ULL;
        R[1] = 0ULL;
        R[2] = 0ULL;
        R[3] = 0ULL;
        R[4] = 0ULL;
        return;
    }

    while (_IsNegative(r)) AddP(r);
    while (!_IsNegative(r)) SubP(r);
    AddP(r);
    Load(R, r);
}

__device__ void _ModMult(uint64_t* r, uint64_t* a, uint64_t* b) {
    uint64_t r512[8];
    uint64_t t[NBBLOCK];
    r512[5] = 0; r512[6] = 0; r512[7] = 0;

    UMult(r512, a, b[0]);
    UMult(t, a, b[1]);
    UADDO1(r512[1], t[0]); UADDC1(r512[2], t[1]); UADDC1(r512[3], t[2]); UADDC1(r512[4], t[3]); UADD1(r512[5], t[4]);
    UMult(t, a, b[2]);
    UADDO1(r512[2], t[0]); UADDC1(r512[3], t[1]); UADDC1(r512[4], t[2]); UADDC1(r512[5], t[3]); UADD1(r512[6], t[4]);
    UMult(t, a, b[3]);
    UADDO1(r512[3], t[0]); UADDC1(r512[4], t[1]); UADDC1(r512[5], t[2]); UADDC1(r512[6], t[3]); UADD1(r512[7], t[4]);

    UMult(t, (r512 + 4), 0x1000003D1ULL);
    UADDO1(r512[0], t[0]); UADDC1(r512[1], t[1]); UADDC1(r512[2], t[2]); UADDC1(r512[3], t[3]);

    uint64_t ah, al;
    UADD1(t[4], 0ULL);
    UMULLO(al, t[4], 0x1000003D1ULL);
    UMULHI(ah, t[4], 0x1000003D1ULL);
    UADDO(r[0], r512[0], al);
    UADDC(r[1], r512[1], ah);
    UADDC(r[2], r512[2], 0ULL);
    UADD(r[3], r512[3], 0ULL);
}

__device__ void _ModMult(uint64_t* r, uint64_t* a) {
    uint64_t r512[8];
    uint64_t t[NBBLOCK];
    uint64_t ah, al;
    r512[5] = 0; r512[6] = 0; r512[7] = 0;

    UMult(r512, a, r[0]);
    UMult(t, a, r[1]);
    UADDO1(r512[1], t[0]); UADDC1(r512[2], t[1]); UADDC1(r512[3], t[2]); UADDC1(r512[4], t[3]); UADD1(r512[5], t[4]);
    UMult(t, a, r[2]);
    UADDO1(r512[2], t[0]); UADDC1(r512[3], t[1]); UADDC1(r512[4], t[2]); UADDC1(r512[5], t[3]); UADD1(r512[6], t[4]);
    UMult(t, a, r[3]);
    UADDO1(r512[3], t[0]); UADDC1(r512[4], t[1]); UADDC1(r512[5], t[2]); UADDC1(r512[6], t[3]); UADD1(r512[7], t[4]);

    UMult(t, (r512 + 4), 0x1000003D1ULL);
    UADDO1(r512[0], t[0]); UADDC1(r512[1], t[1]); UADDC1(r512[2], t[2]); UADDC1(r512[3], t[3]);

    UADD1(t[4], 0ULL);
    UMULLO(al, t[4], 0x1000003D1ULL);
    UMULHI(ah, t[4], 0x1000003D1ULL);
    UADDO(r[0], r512[0], al);
    UADDC(r[1], r512[1], ah);
    UADDC(r[2], r512[2], 0ULL);
    UADD(r[3], r512[3], 0ULL);
}

__device__ void _ModSqr(uint64_t* rp, const uint64_t* up) {
    uint64_t r512[8];
    uint64_t SL, SH;

    {
        uint64_t r01L, r01H, r02L, r02H, r03L, r03H;
        UMULLO(SL, up[0], up[0]); UMULHI(SH, up[0], up[0]);
        UMULLO(r01L, up[0], up[1]); UMULHI(r01H, up[0], up[1]);
        UMULLO(r02L, up[0], up[2]); UMULHI(r02H, up[0], up[2]);
        UMULLO(r03L, up[0], up[3]); UMULHI(r03H, up[0], up[3]);

        r512[0] = SL; r512[1] = r01L; r512[2] = r02L; r512[3] = r03L;
        UADDO1(r512[1], SH); UADDC1(r512[2], r01H); UADDC1(r512[3], r02H); UADD(r512[4], r03H, 0ULL);

        uint64_t r12L, r12H, r13L, r13H;
        UMULLO(SL, up[1], up[1]); UMULHI(SH, up[1], up[1]);
        UMULLO(r12L, up[1], up[2]); UMULHI(r12H, up[1], up[2]);
        UMULLO(r13L, up[1], up[3]); UMULHI(r13H, up[1], up[3]);

        UADDO1(r512[1], r01L); UADDC1(r512[2], SL); UADDC1(r512[3], r12L); UADDC1(r512[4], r13L); UADD(r512[5], r13H, 0ULL);
        UADDO1(r512[2], r01H); UADDC1(r512[3], SH); UADDC1(r512[4], r12H); UADD1(r512[5], 0ULL);

        uint64_t r23L, r23H;
        UMULLO(SL, up[2], up[2]); UMULHI(SH, up[2], up[2]);
        UMULLO(r23L, up[2], up[3]); UMULHI(r23H, up[2], up[3]);

        UADDO1(r512[2], r02L); UADDC1(r512[3], r12L); UADDC1(r512[4], SL); UADDC1(r512[5], r23L); UADD(r512[6], r23H, 0ULL);
        UADDO1(r512[3], r02H); UADDC1(r512[4], r12H); UADDC1(r512[5], SH); UADD1(r512[6], 0ULL);

        UMULLO(SL, up[3], up[3]); UMULHI(SH, up[3], up[3]);
        UADDO1(r512[3], r03L); UADDC1(r512[4], r13L); UADDC1(r512[5], r23L); UADDC1(r512[6], SL); UADD(r512[7], SH, 0ULL);
        UADDO1(r512[4], r03H); UADDC1(r512[5], r13H); UADDC1(r512[6], r23H); UADD1(r512[7], 0ULL);
    }

    uint64_t t[NBBLOCK];
    UMult(t, (r512 + 4), 0x1000003D1ULL);
    UADDO1(r512[0], t[0]); UADDC1(r512[1], t[1]); UADDC1(r512[2], t[2]); UADDC1(r512[3], t[3]);

    UADD1(t[4], 0ULL);
    UMULLO(SL, t[4], 0x1000003D1ULL);
    UMULHI(SH, t[4], 0x1000003D1ULL);
    UADDO(rp[0], r512[0], SL); UADDC(rp[1], r512[1], SH); UADDC(rp[2], r512[2], 0ULL); UADD(rp[3], r512[3], 0ULL);
}

__device__ __noinline__ void _ModInvGrouped(uint64_t r[GPU_GRP_SIZE][4]) {
    uint64_t subp[GPU_GRP_SIZE][4];
    uint64_t newValue[4];
    uint64_t inverse[5];

    Load256(subp[0], r[0]);
    for (uint32_t i = 1; i < GPU_GRP_SIZE; i++) {
        _ModMult(subp[i], subp[i - 1], r[i]);
    }

    Load256(inverse, subp[GPU_GRP_SIZE - 1]);
    inverse[4] = 0;
    _ModInv(inverse);

    for (uint32_t i = GPU_GRP_SIZE - 1; i > 0; i--) {
        _ModMult(newValue, subp[i - 1], inverse);
        _ModMult(inverse, r[i]);
        Load256(r[i], newValue);
    }

    Load256(r[0], inverse);
}

// ---------------------------------------------------------------------------------------
// RHO HELPERS & MODIFICATIONS
// ---------------------------------------------------------------------------------------

__device__ __forceinline__ void AddModOrder256(uint64_t* r, const uint64_t* a) {
    uint64_t t0, t1, t2, t3;
    uint32_t c;

    // 1. Събиране: t = r + a
    UADDO(t0, r[0], a[0]);
    UADDC(t1, r[1], a[1]);
    UADDC(t2, r[2], a[2]);
    UADDC(t3, r[3], a[3]);
    asm volatile ("addc.u32 %0, 0, 0;" : "=r"(c)); // c = 1 ако има пренос извън 256-те бита

    uint64_t s0, s1, s2, s3;
    uint32_t b;

    // 2. Изваждане на модула: s = t - _O
    USUBO(s0, t0, _O[0]);
    USUBC(s1, t1, _O[1]);
    USUBC(s2, t2, _O[2]);
    USUBC(s3, t3, _O[3]);
    asm volatile ("subc.u32 %0, 0, 0;" : "=r"(b)); // b = 1 ако t е по-малко от _O

    // 3. Branchless селекция (Ако има пренос от събирането ИЛИ няма заем от изваждането)
    // Компилаторът NVCC превръща това в хардуерна PTX 'selp' инструкция без if/else!
    bool over = c || (!b);
    r[0] = over ? s0 : t0;
    r[1] = over ? s1 : t1;
    r[2] = over ? s2 : t2;
    r[3] = over ? s3 : t3;
}

__device__ void LoadKangaroos(uint64_t* a_mem, uint64_t px[GPU_GRP_SIZE][4], uint64_t py[GPU_GRP_SIZE][4], uint64_t a_scal[GPU_GRP_SIZE][4], uint64_t b_scal[GPU_GRP_SIZE][4]) {
    __syncthreads();
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * blockDim.x;
        for (int i = 0; i < 4; i++) {
            px[g][i] = a_mem[IDX + (0 + i) * blockDim.x + stride];
            py[g][i] = a_mem[IDX + (4 + i) * blockDim.x + stride];
            a_scal[g][i] = a_mem[IDX + (8 + i) * blockDim.x + stride];
            b_scal[g][i] = a_mem[IDX + (12 + i) * blockDim.x + stride];
        }
    }
}

__device__ void StoreKangaroos(uint64_t* a_mem, uint64_t px[GPU_GRP_SIZE][4], uint64_t py[GPU_GRP_SIZE][4], uint64_t a_scal[GPU_GRP_SIZE][4], uint64_t b_scal[GPU_GRP_SIZE][4]) {
    __syncthreads();
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * blockDim.x;
        for (int i = 0; i < 4; i++) {
            a_mem[IDX + (0 + i) * blockDim.x + stride] = px[g][i];
            a_mem[IDX + (4 + i) * blockDim.x + stride] = py[g][i];
            a_mem[IDX + (8 + i) * blockDim.x + stride] = a_scal[g][i];
            a_mem[IDX + (12 + i) * blockDim.x + stride] = b_scal[g][i];
        }
    }
}

// ===========================================================================
// ДОБАВИ ТОВА В GPUMath.h, НЕПОСРЕДСТВЕНО ПРЕДИ  #endif // GPUMATHH
//
// Замества старите LoadKangaroos / StoreKangaroos с Ро-варианти.
// Старите може да останат -- вече не се използват.
// ===========================================================================

// ---------------------------------------------------------------------------
// Мащабирана таблица скокове: jCG[e][i] = lambda^(-e) * c_i (mod n)
// Пълни се от хоста в GPUEngine::SetParams
// ---------------------------------------------------------------------------
__device__ __constant__ uint64_t jCG[3][NB_JUMP][4];

// KSIZE за Ро: 4 (px) + 4 (py) + 4 (A) + 1 (state) = 13
// Дефинира се в GPUEngine.h, тук само се използва.

// ---------------------------------------------------------------------------
// A = A - b  (mod n)   -- реда на групата, НЕ полето
// ---------------------------------------------------------------------------
__device__ __forceinline__ void SubModOrder256(uint64_t* r, const uint64_t* a) {
    uint64_t t0, t1, t2, t3;
    uint32_t borrow;

    USUBO(t0, r[0], a[0]);
    USUBC(t1, r[1], a[1]);
    USUBC(t2, r[2], a[2]);
    USUBC(t3, r[3], a[3]);
    asm volatile ("subc.u32 %0, 0, 0;" : "=r"(borrow));

    // Ако е имало заем -> добавяме модула обратно
    uint64_t m = (borrow != 0) ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
    uint64_t c0 = _O[0] & m;
    uint64_t c1 = _O[1] & m;
    uint64_t c2 = _O[2] & m;
    uint64_t c3 = _O[3] & m;

    UADDO1(t0, c0);
    UADDC1(t1, c1);
    UADDC1(t2, c2);
    UADD1(t3, c3);

    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

// ---------------------------------------------------------------------------
// Зареждане / записване на състоянието на Ро разходките.
// Layout в глобалната памет (coalesced, по 64-битови слотове):
//   слотове  0-3  : px
//   слотове  4-7  : py
//   слотове  8-11 : A
//   слот     12   : state (e | neg | b | tIdx), опакован в долните 32 бита
// ---------------------------------------------------------------------------
__device__ void LoadKangaroosRho(uint64_t* a_mem,
    uint64_t px[GPU_GRP_SIZE][4],
    uint64_t py[GPU_GRP_SIZE][4],
    uint64_t a_scal[GPU_GRP_SIZE][4],
    uint32_t st[GPU_GRP_SIZE]) {
    __syncthreads();
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * blockDim.x;
        for (int i = 0; i < 4; i++) {
            px[g][i] = a_mem[IDX + (0 + i) * blockDim.x + stride];
            py[g][i] = a_mem[IDX + (4 + i) * blockDim.x + stride];
            a_scal[g][i] = a_mem[IDX + (8 + i) * blockDim.x + stride];
        }
        st[g] = (uint32_t)a_mem[IDX + 12 * blockDim.x + stride];
    }
}

__device__ void StoreKangaroosRho(uint64_t* a_mem,
    uint64_t px[GPU_GRP_SIZE][4],
    uint64_t py[GPU_GRP_SIZE][4],
    uint64_t a_scal[GPU_GRP_SIZE][4],
    uint32_t st[GPU_GRP_SIZE]) {
    __syncthreads();
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * blockDim.x;
        for (int i = 0; i < 4; i++) {
            a_mem[IDX + (0 + i) * blockDim.x + stride] = px[g][i];
            a_mem[IDX + (4 + i) * blockDim.x + stride] = py[g][i];
            a_mem[IDX + (8 + i) * blockDim.x + stride] = a_scal[g][i];
        }
        a_mem[IDX + 12 * blockDim.x + stride] = (uint64_t)st[g];
    }
}

#endif // GPUMATHH