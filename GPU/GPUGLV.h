#ifndef GPUGLVH
#define GPUGLVH

// ---------------------------------------------------------------------------
// GLV ендоморфизъм за secp256k1 + negation map
//
// Върху secp256k1 съществува ефективно изчислим ендоморфизъм:
//     phi(x,y) = (beta*x, y)      и      phi(P) = lambda*P
// където beta е кубичен корен от 1 по модул p, а lambda е кубичен корен
// от 1 по модул n (реда на групата).
//
// Заедно с negation (x,y) -> (x,-y) това дава 6-орна симетрия:
//     { P, -P, phi(P), -phi(P), phi^2(P), -phi^2(P) }
// Три различни x-координати: { x, beta*x, beta^2*x }
//
// Свеждайки всяка точка до каноничен представител, пространството за
// търсене се свива 6 пъти -> sqrt(3) ~ 1.73x спрямо чист negation (sqrt(2)).
//
// ВНИМАНИЕ: 6-орната симетрия прави безплодните цикли ЗНАЧИТЕЛНО по-чести.
// Ако escape логиката се задейства често, цената надхвърля печалбата.
// Затова има превключвател NEGATION_ONLY (дефинира се в Constants.h).
//
// Скаларите: ако R = phi^e(+-(A*G + B*T)), то
//     a_true = s * lambda^e * A   (mod n)
//     b_true = s * lambda^e * B   (mod n)
// Корекцията се прилага ВЕДНЪЖ на CPU при колизия, не на всяка стъпка.
// В разходката A се обновява чрез предварително мащабирана таблица
//     jCG[e][i] = lambda^(-e) * c_i (mod n)
// -> нула модулни умножения на стъпка.
// ---------------------------------------------------------------------------

#include <stdint.h>
#include <cuda_runtime.h>

// beta = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
__device__ __constant__ uint64_t _BETA[4] = {
    0xc1396c28719501eeULL, 0x9cf0497512f58995ULL,
    0x6e64479eac3434e9ULL, 0x7ae96a2b657c0710ULL
};

// beta^2 mod p = 0x851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40
__device__ __constant__ uint64_t _BETA2[4] = {
    0x3ec693d68e6afa40ULL, 0x630fb68aed0a766aULL,
    0x919bb86153cbcb16ULL, 0x851695d49a83f8efULL
};

// (p-1)/2 -- за каноничен избор на знака на y
__device__ __constant__ uint64_t _PHALF[4] = {
    0xffffffff7ffffe17ULL, 0xffffffffffffffffULL,
    0xffffffffffffffffULL, 0x7fffffffffffffffULL
};

// p -- за отрицание на y
__device__ __constant__ uint64_t _PFIELD[4] = {
    0xfffffffefffffc2fULL, 0xffffffffffffffffULL,
    0xffffffffffffffffULL, 0xffffffffffffffffULL
};

// ---------------------------------------------------------------------------
// Сравнение: 1 ако a > b, -1 ако a < b, 0 ако равни
// ---------------------------------------------------------------------------
__device__ __forceinline__ int _Cmp256(const uint64_t* a, const uint64_t* b) {
    if (a[3] != b[3]) return (a[3] > b[3]) ? 1 : -1;
    if (a[2] != b[2]) return (a[2] > b[2]) ? 1 : -1;
    if (a[1] != b[1]) return (a[1] > b[1]) ? 1 : -1;
    if (a[0] != b[0]) return (a[0] > b[0]) ? 1 : -1;
    return 0;
}

// ---------------------------------------------------------------------------
// y -> p - y
// ---------------------------------------------------------------------------
__device__ __forceinline__ void _NegModP(uint64_t* y) {
    uint64_t t[4];
    USUBO(t[0], _PFIELD[0], y[0]);
    USUBC(t[1], _PFIELD[1], y[1]);
    USUBC(t[2], _PFIELD[2], y[2]);
    USUB(t[3], _PFIELD[3], y[3]);
    y[0] = t[0]; y[1] = t[1]; y[2] = t[2]; y[3] = t[3];
}

// ---------------------------------------------------------------------------
// Каноничен представител на класа.
//
// Изход: px, py -- каноничната точка
//        *outE   -- 0/1/2  приложена степен на phi  (винаги 0 при NEGATION_ONLY)
//        *outNeg -- 0/1    приложено ли е отрицание
// ---------------------------------------------------------------------------
__device__ __forceinline__ void _CanonicalClass(uint64_t* px, uint64_t* py,
                                                uint32_t* outE, uint32_t* outNeg) {

#ifdef NEGATION_ONLY

    // Клас { P, -P } -- свиване 2 пъти, sqrt(2).
    // Безплодните цикли са много по-редки, отколкото при 6-орна симетрия.
    *outE = 0;
    uint32_t neg = 0;
    if (_Cmp256(py, (uint64_t*)_PHALF) > 0) {
        _NegModP(py);
        neg = 1;
    }
    *outNeg = neg;

#else

    // Клас { +-P, +-phi(P), +-phi^2(P) } -- свиване 6 пъти, sqrt(3).
    uint64_t x1[4], x2[4];

    _ModMult(x1, px, (uint64_t*)_BETA);
    _ModMult(x2, px, (uint64_t*)_BETA2);

    uint32_t e = 0;
    if (_Cmp256(x1, px) < 0) e = 1;
    if (e == 0) { if (_Cmp256(x2, px) < 0) e = 2; }
    else        { if (_Cmp256(x2, x1)  < 0) e = 2; }

    if (e == 1) { Load256(px, x1); }
    else if (e == 2) { Load256(px, x2); }

    uint32_t neg = 0;
    if (_Cmp256(py, (uint64_t*)_PHALF) > 0) {
        _NegModP(py);
        neg = 1;
    }

    *outE = e;
    *outNeg = neg;

#endif
}

#endif // GPUGLVH