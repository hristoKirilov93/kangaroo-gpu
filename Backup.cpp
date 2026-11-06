/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* (Modified for Pollard's Rho - Backup features disabled)
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
#include <sys/stat.h>
#endif

using namespace std;


// ----------------------------------------------------------------------------

int Kangaroo::FSeek(FILE* stream, uint64_t pos) {

#ifdef WIN64
    return _fseeki64(stream, pos, SEEK_SET);
#else
    return fseeko(stream, pos, SEEK_SET);
#endif

}

uint64_t Kangaroo::FTell(FILE* stream) {

#ifdef WIN64
    return (uint64_t)_ftelli64(stream);
#else
    return (uint64_t)ftello(stream);
#endif

}

bool Kangaroo::IsEmpty(std::string fileName) {
    FILE* pFile = fopen(fileName.c_str(), "r");
    if (pFile == NULL) {
        ::printf("OpenPart: Cannot open %s for reading\n", fileName.c_str());
        ::printf("%s\n", ::strerror(errno));
        ::exit(0);
    }
    fseek(pFile, 0, SEEK_END);
    uint32_t size = ftell(pFile);
    fclose(pFile);
    return size == 0;
}

int Kangaroo::IsDir(string dirName) {
    bool isDir = 0;
#ifdef WIN64
    WIN32_FIND_DATA ffd;
    HANDLE hFind;
    hFind = FindFirstFile(dirName.c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE) {
        ::printf("%s not found\n", dirName.c_str());
        return -1;
    }
    isDir = (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    FindClose(hFind);
#else
    struct stat buffer;
    if (stat(dirName.c_str(), &buffer) != 0) {
        ::printf("%s not found\n", dirName.c_str());
        return -1;
    }
    isDir = (buffer.st_mode & S_IFDIR) != 0;
#endif
    return isDir;
}

FILE* Kangaroo::ReadHeader(std::string fileName, uint32_t* version, int type) {
    return NULL; // Disabled for Rho
}

bool Kangaroo::LoadWork(string& fileName) {
    ::printf("LoadWork is disabled for Rho.\n");
    return false;
}

// ----------------------------------------------------------------------------

void Kangaroo::FetchWalks(uint64_t nbWalk, Int* x, Int* y, Int* a, Int* b) {
    // Disabled for Rho
}

void Kangaroo::FetchWalks(uint64_t nbWalk, std::vector<int128_t>& kangs, Int* x, Int* y, Int* a, Int* b) {
    // Disabled for Rho
}

void Kangaroo::FectchKangaroos(TH_PARAM* threads) {
    // Disabled for Rho. Just create fresh kangaroos instead of loading.
    ::printf("FetchKangaroos is disabled. Creating fresh random kangaroos for Rho...\n");
}


// ----------------------------------------------------------------------------

bool Kangaroo::SaveHeader(string fileName, FILE* f, int type, uint64_t totalCount, double totalTime) {
    return false; // Disabled for Rho
}

void  Kangaroo::SaveWork(string fileName, FILE* f, int type, uint64_t totalCount, double totalTime) {
    // Disabled for Rho
}

void Kangaroo::SaveServerWork() {
    saveRequest = false; // Disabled for Rho
}

void Kangaroo::SaveWork(uint64_t totalCount, double totalTime, TH_PARAM* threads, int nbThread) {
    // Disabled for Rho
    saveRequest = false;
    UNLOCK(saveMutex);
}

void Kangaroo::WorkInfo(std::string& fName) {
    ::printf("WorkInfo is disabled for Rho.\n");
}