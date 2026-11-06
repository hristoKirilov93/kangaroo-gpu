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
#include <signal.h>
#ifndef WIN64
#include <pthread.h>
#else
#include "WindowsErrors.h"
#endif

using namespace std;

static SOCKET serverSock = 0;

// ------------------------------------------------------------------------------------------------------
// Common part
// ------------------------------------------------------------------------------------------------------

#define MAX_CLIENT 256
#define WAIT_FOR_READ  1
#define WAIT_FOR_WRITE 2

#define SERVER_VERSION 3

#define SERVER_HEADER 0x67DEDDC1

#define KANG_PER_BLOCK 2048

// Commands
#define SERVER_GETCONFIG 0
#define SERVER_STATUS    1
#define SERVER_SENDDP    2
#define SERVER_SETKNB    3
#define SERVER_SAVEKANG  4
#define SERVER_LOADKANG  5
#define SERVER_RESETDEAD  'R'

// Status
#define SERVER_OK            0
#define SERVER_END           1
#define SERVER_BACKUP        2


#ifdef WIN64

#define close_socket(s) closesocket(s)

typedef int socklen_t;

string GetNetworkError() {

    int err = WSAGetLastError();
    bool found = false;
    int i = 0;
    while (!found && i < NBWSAERRORS) {
        found = (WSAERRORS[i].errCode == err);
        if (!found) i++;
    }

    if (!found) {
        char ret[256];
        sprintf(ret, "WSA Error code %d", err);
        return string(ret);
    }
    else {
        return string(WSAERRORS[i].mesg);
    }

}

#else

#define close_socket(s) close(s)

string GetNetworkError() {

    return string(strerror(errno));

}

#endif

#define GET(name,s,b,bl,t)  if( (nbRead=Read(s,(char *)(b),bl,t))<0 ) { ::printf("\nReadError(" name "): %s\n",lastError.c_str()); isConnected = false; close_socket(s); return false; }
#define PUT(name,s,b,bl,t)  if( (nbWrite=Write(s,(char *)(b),bl,t))<0 ) { ::printf("\nWriteError(" name "): %s\n",lastError.c_str()); isConnected = false; close_socket(s); return false; }
#define GETFREE(name,s,b,bl,t,x)  if( (nbRead=Read(s,(char *)(b),bl,t))<0 ) { ::printf("\nReadError(" name "): %s\n",lastError.c_str()); isConnected = false; ::free(x); close_socket(s); return false; }
#define PUTFREE(name,s,b,bl,t,x)  if( (nbWrite=Write(s,(char *)(b),bl,t))<0 ) { ::printf("\nWriteError(" name "): %s\n",lastError.c_str()); isConnected = false; ::free(x); close_socket(s); return false; }

void sig_handler(int signo) {
    if (signo == SIGINT) {
        ::printf("\nTerminated\n");
        if (serverSock > 0) close_socket(serverSock);
#ifdef WIN64
        WSACleanup();
#endif
        exit(0);
    }
}

int Kangaroo::WaitFor(SOCKET sock, int timeout, int mode) {

    fd_set fdset;
    fd_set* rd = NULL, * wr = NULL;
    struct timeval tmout;
    int result;

    FD_ZERO(&fdset);
    FD_SET(sock, &fdset);
    if (mode == WAIT_FOR_READ)
        rd = &fdset;
    if (mode == WAIT_FOR_WRITE)
        wr = &fdset;

    tmout.tv_sec = (int)(timeout / 1000);
    tmout.tv_usec = (int)(timeout % 1000) * 1000;

    do
        result = select((int)sock + 1, rd, wr, NULL, &tmout);
#ifdef WIN64
    while (result < 0 && WSAGetLastError() == WSAEINTR);
#else
    while (result < 0 && errno == EINTR);
#endif

    if (result == 0) {
        lastError = "The operation timed out";
    }
    else if (result < 0) {
        lastError = GetNetworkError();
        return 0;
    }

    return result;

}

int Kangaroo::Write(SOCKET sock, char* buf, int bufsize, int timeout) {

    int total_written = 0;
    int written = 0;

    while (bufsize > 0)
    {
        // Wait
        if (!WaitFor(sock, timeout, WAIT_FOR_WRITE))
            return -1;

        // Write
        do
            written = send(sock, buf, bufsize, 0);
#ifdef WIN64
        while (written == -1 && WSAGetLastError() == WSAEINTR);
#else
        while (written == -1 && errno == EINTR);
#endif

        if (written <= 0)
            break;

        buf += written;
        total_written += written;
        bufsize -= written;
    }

    if (written < 0) {
        lastError = GetNetworkError();
        return -1;
    }

    if (bufsize != 0) {
        lastError = "Failed to send entire buffer";
        return -1;
    }

    return total_written;

}

int Kangaroo::Read(SOCKET sock, char* buf, int bufsize, int timeout) { // Timeout in millisec

    int rd = 0;
    int total_read = 0;

    while (bufsize > 0) {

        // Wait
        if (!WaitFor(sock, timeout, WAIT_FOR_READ)) {
            return -1;
        }

        // Read
        do
            rd = recv(sock, buf, bufsize, 0);
#ifdef WIN64
        while (rd == -1 && WSAGetLastError() == WSAEINTR);
#else
        while (rd == -1 && errno == EINTR);
#endif
        if (rd <= 0)
            break;

        buf += rd;
        total_read += rd;
        bufsize -= rd;

    }

    if (rd < 0) {
        lastError = GetNetworkError();
        return -1;
    }

    if (rd == 0) {
        lastError = "Connection closed";
        return -1;
    }

    return total_read;

}

void Kangaroo::InitSocket() {

#ifdef WIN64
    // connect to Winscok DLL
    WSADATA WSAData;
    int err = WSAStartup(MAKEWORD(2, 2), &WSAData);
    if (err != 0) {
        ::printf("WSAStartup failed error : %d\n", err);
        exit(-1);
    }
#endif

}

// Server status
int32_t Kangaroo::GetServerStatus() {

    if (endOfSearch) {
        return SERVER_END;
    }

    if (saveRequest) {
        return SERVER_BACKUP;
    }

    return SERVER_OK;

}

#define CLIENT_ABORT() \
::printf("\nClosing connection with %s\n",p->clientInfo); \
close_socket(p->clientSock); \
return false;

// Server request handler
bool Kangaroo::HandleRequest(TH_PARAM* p) {
    // Функцията остава същата, но коментираме SERVER_SENDDP логиката
    char cmdBuff;
    uint32_t version = SERVER_VERSION;
    int nbRead;
    int nbWrite;
    int32_t state;

    while (p->isRunning) {
        nbRead = Read(p->clientSock, (char*)(&cmdBuff), 1, (int)(CLIENT_TIMEOUT * 1000.0));
        if (nbRead <= 0) { CLIENT_ABORT(); }
        switch (cmdBuff) {
        case SERVER_GETCONFIG: {
            ::printf("\nNew connection from %s\n", p->clientInfo);
            PUT("Version", p->clientSock, &version, sizeof(uint32_t), ntimeout);
            PUT("RangeStart", p->clientSock, rangeStart.bits64, 32, ntimeout);
            PUT("RangeEnd", p->clientSock, rangeEnd.bits64, 32, ntimeout);
            PUT("KeyX", p->clientSock, keysToSearch[keyIdx].x.bits64, 32, ntimeout);
            PUT("KeyY", p->clientSock, keysToSearch[keyIdx].y.bits64, 32, ntimeout);
            PUT("DP", p->clientSock, &initDPSize, sizeof(int32_t), ntimeout);
        } break;
        case SERVER_SETKNB: {
            GET("nbKangaroo", p->clientSock, &p->nbKangaroo, sizeof(uint64_t), ntimeout);
            totalRW += p->nbKangaroo;
        } break;
        case SERVER_RESETDEAD: {
            char response[5];
            collisionInSameHerd = 0;
            GET("flush", p->clientSock, &response, 2, ntimeout);
            sprintf(response, "OK\n");
            PUT("resp", p->clientSock, &response, 3, ntimeout);
        } break;
        case SERVER_LOADKANG: {
            // Игнорираме за Rho
            uint64_t nbKangaroo = 0;
            PUT("nbKangaroo", p->clientSock, &nbKangaroo, sizeof(uint64_t), ntimeout);
        } break;
        case SERVER_SAVEKANG: {
            // Игнорираме за Rho
        } break;
        case SERVER_STATUS: {
            state = GetServerStatus();
            PUT("Status", p->clientSock, &state, sizeof(int32_t), ntimeout);
        } break;
        case SERVER_SENDDP: {
            // Disabled for Rho Multi-Target network protocol
            ::printf("\nNetwork DP Send/Receive is disabled for Rho.\n");
            CLIENT_ABORT();
        } break;
        default:
            ::printf("\nUnexpected command [%d] from %s\n", cmdBuff, p->clientInfo);
            CLIENT_ABORT();
        }
    }
    close_socket(p->clientSock);
    return true;
}

// Threaded proc
#ifdef WIN64
DWORD WINAPI _acceptThread(LPVOID lpParam) {
#else
void* _acceptThread(void* lpParam) {
#endif
    TH_PARAM* p = (TH_PARAM*)lpParam;
    p->obj->AddConnectedClient();
    p->obj->HandleRequest(p);
    p->obj->RemoveConnectedClient();
    p->obj->RemoveConnectedKangaroo(p->nbKangaroo);
    p->isRunning = false;
    free(p->clientInfo);
    free(p);
    return 0;
}

#ifdef WIN64
DWORD WINAPI _processServer(LPVOID lpParam) {
#else
void* _processServer(void* lpParam) {
#endif
    Kangaroo* obj = (Kangaroo*)lpParam;
    obj->ProcessServer();
    return 0;
}

// Main server loop
void Kangaroo::AcceptConnections(SOCKET server_soc) {
    SOCKET clientSock;
    ::printf("Kangaroo server is ready and listening to TCP port %d ...\n", port);
    while (true) {
        struct sockaddr_in client_add;
        socklen_t len = sizeof(sockaddr_in);
        if ((clientSock = accept(server_soc, (struct sockaddr*)&client_add, &len)) < 0) {
            ::printf("Error: Invalid Socket returned by accept(): %s\n", GetNetworkError().c_str());
        }
        else {
            TH_PARAM* p = (TH_PARAM*)malloc(sizeof(TH_PARAM));
            ::memset(p, 0, sizeof(TH_PARAM));
            char info[256];
            ::sprintf(info, "%s:%d", inet_ntoa(client_add.sin_addr), ntohs(client_add.sin_port));
#ifdef WIN64
            p->clientInfo = ::_strdup(info);
#else
            p->clientInfo = ::strdup(info);
#endif
            p->obj = this;
            p->isRunning = true;
            p->clientSock = clientSock;
            LaunchThread(_acceptThread, p);
        }
    }
}

// Starts the server
void Kangaroo::RunServer() {
    ::printf("Server mode is currently unsupported in this Rho build.\n");
    exit(0);
}

// ------------------------------------------------------------------------------------------------------
// Client part
// ------------------------------------------------------------------------------------------------------

bool Kangaroo::ConnectToServer(SOCKET * retSock) {
    return false; // Disabled
}

void Kangaroo::WaitForServer() {
    // Disabled
}

bool Kangaroo::GetKangaroosFromServer(std::string & fileName, std::vector<int128_t>&kangs) {
    return false;
}

bool Kangaroo::SendKangaroosToServer(std::string & fileName, std::vector<int128_t>&kangs) {
    return false;
}

bool Kangaroo::SendToServer(std::vector<ITEM> &dps, uint32_t threadId, uint32_t gpuId) {
    // Disabled for Rho
    return true;
}

void Kangaroo::AddConnectedClient() {
    connectedClient++;
}

void Kangaroo::RemoveConnectedClient() {
    connectedClient--;
}

void Kangaroo::RemoveConnectedKangaroo(uint64_t nb) {
    totalRW -= nb;
}

bool Kangaroo::GetConfigFromServer() {
    return false;
}