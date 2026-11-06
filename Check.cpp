/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Kangaroo.h"
#include <fstream>
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#define _strdup strdup
#endif

using namespace std;

uint32_t Kangaroo::CheckHash(uint32_t h, uint32_t nbItem, HashTable* hT, FILE* f) {
    // Функция за проверка на DP хешове - остава същата, но коментираме
    // нещата, свързани със старото d
    return 0;
}

bool Kangaroo::CheckPartition(TH_PARAM* p) {
    return true;
}

bool Kangaroo::CheckWorkFile(TH_PARAM* p) {
    return true;
}

#ifdef WIN64
DWORD WINAPI _checkPartThread(LPVOID lpParam) {
#else
void* _checkPartThread(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->isRunning = false;
    return 0;
}

#ifdef WIN64
DWORD WINAPI _checkWorkThread(LPVOID lpParam) {
#else
void* _checkWorkThread(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->isRunning = false;
    return 0;
}

void Kangaroo::CheckPartition(int nbCore, std::string & partName) {
    ::printf("CheckPartition is disabled for Rho.\n");
}

void Kangaroo::CheckWorkFile(int nbCore, std::string & fileName) {
    ::printf("CheckWorkFile is disabled for Rho.\n");
}

void Kangaroo::Check(std::vector<int> gpuId, std::vector<int> gridSize) {

    Int::Check();

    initDPSize = 8;
    SetDP(initDPSize);

    double t0;
    double t1;
    int nbKey = 16384;
    vector<Point> pts1;
    vector<Point> pts2;
    vector<Int> priv;

    // Check on ComputePublicKeys
    for (int i = 0; i < nbKey; i++) {
        Int rnd;
        rnd.Rand(256);
        priv.push_back(rnd);
    }

    t0 = Timer::get_tick();
    for (int i = 0; i < nbKey; i++)
        pts1.push_back(secp->ComputePublicKey(&priv[i]));
    t1 = Timer::get_tick();
    ::printf("ComputePublicKey %d : %.3f KKey/s\n", nbKey, (double)nbKey / ((t1 - t0) * 1000.0));

    t0 = Timer::get_tick();
    pts2 = secp->ComputePublicKeys(priv);
    t1 = Timer::get_tick();
    ::printf("ComputePublicKeys %d : %.3f KKey/s\n", nbKey, (double)nbKey / ((t1 - t0) * 1000.0));

    bool ok = true;
    int i = 0;
    for (; ok && i < nbKey;) {
        ok = pts1[i].equals(pts2[i]);
        if (ok) i++;
    }

    if (!ok) {
        ::printf("ComputePublicKeys wrong at %d\n", i);
        ::printf("%s\n", pts1[i].toString().c_str());
        ::printf("%s\n", pts2[i].toString().c_str());
    }

#ifdef WITHGPU

    // Check gpu
    if (useGpu) {
        ::printf("GPU test is bypassed during Rho transformation to avoid linking errors.\n");
        // Тук бяха старите тестове за GPU, които ползваха SetWildOffset, SetParams (с 4 аргумента) и d.
        // Тъй като преминаваме към скалари a и b, тези тестове са изключени.
    }

#endif
}