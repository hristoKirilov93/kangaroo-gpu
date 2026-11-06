/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* (Modified for Pollard's Rho - PartMerge features disabled)
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
#include <dirent.h>
#include <pthread.h>
#include <sys/stat.h>
#define _strdup strdup
#endif

using namespace std;

string Kangaroo::GetPartName(std::string& partName, int i, bool tmpPart) {
	return "";
}

FILE* Kangaroo::OpenPart(std::string& partName, char* mode, int i, bool tmpPart) {
	return NULL;
}

void Kangaroo::CreateEmptyPartWork(std::string& partName) {
	::printf("CreateEmptyPartWork is disabled for Rho.\n");
}

bool Kangaroo::MergePartition(TH_PARAM* p) {
	::printf("MergePartition is disabled for Rho.\n");
	return false;
}

// Threaded proc
#ifdef WIN64
DWORD WINAPI _mergeThread(LPVOID lpParam) { return 0; }
#else
void* _mergeThread(void* lpParam) { return 0; }
#endif

// Threaded proc
#ifdef WIN64
DWORD WINAPI _mergePartThread(LPVOID lpParam) { return 0; }
#else
void* _mergePartThread(void* lpParam) { return 0; }
#endif

bool Kangaroo::MergeWorkPartPart(std::string& part1Name, std::string& part2Name) {
	::printf("MergeWorkPartPart is disabled for Rho.\n");
	return false;
}

bool Kangaroo::FillEmptyPartFromFile(std::string& partName, std::string& fileName, bool printStat) {
	::printf("FillEmptyPartFromFile is disabled for Rho.\n");
	return false;
}

bool Kangaroo::MergeWorkPart(std::string& partName, std::string& file2, bool printStat) {
	::printf("MergeWorkPart is disabled for Rho.\n");
	return false;
}