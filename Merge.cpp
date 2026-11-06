/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* (Modified for Pollard's Rho - Merge features disabled)
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
#endif

using namespace std;

bool Kangaroo::MergeWork(std::string& file1, std::string& file2, std::string& dest, bool printStat) {
	::printf("MergeWork is disabled for Rho.\n");
	return false;
}

typedef struct File {
	std::string name;
	uint64_t size;
} File;

bool sortBySize(const File& lhs, const File& rhs) { return lhs.size > rhs.size; }

void Kangaroo::MergeDir(std::string& dirName, std::string& dest) {
	::printf("MergeDir is disabled for Rho.\n");
}