/** This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*/

#ifndef CONSTANTSH
#define CONSTANTSH

//#define NEGATION_ONLY

// Release number
#define RELEASE "8"

// Number of random jumps
#define NB_JUMP 256

// GPU group size
#define GPU_GRP_SIZE 20

// GPU number of run per kernel call (Увеличено от 64 на 512 за максимален MK/s)
#define NB_RUN 1024

// Kangaroo type
#define TAME 0  // Tame kangaroo
#define WILD 1  // Wild kangaroo

// SendDP Period in sec
#define SEND_PERIOD 2.0

// Timeout before closing connection idle client in sec
#define CLIENT_TIMEOUT 3600.0

// Number of merge partition
#define MERGE_PART 256

#endif //CONSTANTSH