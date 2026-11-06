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

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "Timer.h"

static const char* prefix[] = { "","Kilo","Mega","Giga","Tera","Peta","Hexa" };

#ifdef WIN64
#include <windows.h>
#include <wincrypt.h>
LARGE_INTEGER Timer::perfTickStart;
double Timer::perfTicksPerSec;
LARGE_INTEGER Timer::qwTicksPerSec;
#else
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
time_t Timer::tickStart;
#endif

void Timer::Init() {
#ifdef WIN64
    QueryPerformanceFrequency(&qwTicksPerSec);
    QueryPerformanceCounter(&perfTickStart);
    perfTicksPerSec = (double)qwTicksPerSec.QuadPart;
#else
    tickStart = time(NULL);
#endif
}

double Timer::get_tick() {
#ifdef WIN64
    LARGE_INTEGER t, dt;
    QueryPerformanceCounter(&t);
    dt.QuadPart = t.QuadPart - perfTickStart.QuadPart;
    return (double)(dt.QuadPart) / perfTicksPerSec;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)(tv.tv_sec - tickStart) + (double)tv.tv_usec / 1e6;
#endif
}

uint32_t Timer::getSeed32() {
    return ::strtoul(getSeed(4).c_str(), NULL, 16);
}

uint32_t Timer::getPID() {
#ifdef WIN64
    return (uint32_t)GetCurrentProcessId();
#else
    return (uint32_t)getpid();
#endif
}

std::string Timer::getSeed(int size) {
    std::string ret;
    char tmp[3];
    unsigned char* buff = (unsigned char*)malloc(size);

#ifdef WIN64
    HCRYPTPROV hCryptProv = NULL;
    LPCSTR UserName = "KeyContainer";

    if (!CryptAcquireContext(&hCryptProv, UserName, NULL, PROV_RSA_FULL, 0)) {
        if (GetLastError() == (DWORD)NTE_BAD_KEYSET) {
            if (!CryptAcquireContext(&hCryptProv, UserName, NULL, PROV_RSA_FULL, CRYPT_NEWKEYSET)) {
                printf("CryptAcquireContext(): Could not create a new key container.\n");
                exit(1);
            }
        }
        else {
            printf("CryptAcquireContext(): A cryptographic service handle could not be acquired.\n");
            exit(1);
        }
    }

    if (!CryptGenRandom(hCryptProv, size, buff)) {
        printf("CryptGenRandom(): Error during random sequence acquisition.\n");
        exit(1);
    }

    CryptReleaseContext(hCryptProv, 0);
#else
    FILE* f = fopen("/dev/urandom", "rb");
    if (f == NULL) {
        printf("Failed to open /dev/urandom %s\n", strerror(errno));
        exit(1);
    }
    if (fread(buff, 1, size, f) != (size_t)size) {
        printf("Failed to read from /dev/urandom %s\n", strerror(errno));
        exit(1);
    }
    fclose(f);
#endif

    for (int i = 0; i < size; i++) {
        sprintf(tmp, "%02X", buff[i]);
        ret.append(tmp);
    }

    free(buff);
    return ret;
}

std::string Timer::getResult(char* unit, int nbTry, double t0, double t1) {
    char tmp[256];
    int pIdx = 0;
    double nbCallPerSec = (double)nbTry / (t1 - t0);
    while (nbCallPerSec > 1000.0 && pIdx < 5) {
        pIdx++;
        nbCallPerSec = nbCallPerSec / 1000.0;
    }
    sprintf(tmp, "%.3f %s%s/sec", nbCallPerSec, prefix[pIdx], unit);
    return std::string(tmp);
}

void Timer::printResult(char* unit, int nbTry, double t0, double t1) {
    printf("%s", getResult(unit, nbTry, t0, t1).c_str());
}

int Timer::getCoreNumber() {
#ifdef WIN64
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int)sysinfo.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

void Timer::SleepMillis(uint32_t millis) {
#ifdef WIN64
    Sleep(millis);
#else
    usleep(millis * 1000);
#endif
}

std::string Timer::getTS() {
    std::string ret;
    time_t now = time(NULL);
    char* t = ctime(&now);

    if (t[8] == ' ') t[8] = '0';
    ret.push_back(t[8]);
    ret.push_back(t[9]);
    ret.push_back(t[4]);
    ret.push_back(t[5]);
    ret.push_back(t[6]);
    ret.push_back(t[22]);
    ret.push_back(t[23]);
    ret.push_back('_');
    ret.push_back(t[11]);
    ret.push_back(t[12]);
    ret.push_back(t[14]);
    ret.push_back(t[15]);
    ret.push_back(t[17]);
    ret.push_back(t[18]);

    return ret;
}