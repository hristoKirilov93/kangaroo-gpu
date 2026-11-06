#ifndef KANGAROO_H
#define KANGAROO_H

#ifdef WIN64
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#else
typedef int SOCKET;
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#endif

#include <string>
#include <vector>
#include <unordered_set>
#include "SECPK1/SECP256k1.h"
#include "HashTable.h"
#include "SECPK1/IntGroup.h"
#include "GPU/GPUEngine.h"

#ifdef WIN64
typedef HANDLE THREAD_HANDLE;
#define LOCK(mutex) WaitForSingleObject(mutex, INFINITE);
#define UNLOCK(mutex) ReleaseMutex(mutex);
#else
typedef pthread_t THREAD_HANDLE;
#define LOCK(mutex)  pthread_mutex_lock(&(mutex));
#define UNLOCK(mutex) pthread_mutex_unlock(&(mutex));
#endif

// ===========================================================================
// РЕЖИМ НА РАБОТА
//
// DEMO_BOUNDED  -- Kangaroo в ограничен диапазон. Намира реален ключ за
//                  секунди. Използвай за демонстрация пред лектора.
//                  GLV е ИЗКЛЮЧЕН (в ограничен диапазон няма смисъл --
//                  канонизацията изкарва разходката извън диапазона).
//
// (по подразбиране) -- Ро в целия 256-битов диапазон с GLV.
//                  Архитектурно правилно, но НИКОГА няма да намери ключ.
//                  Използвай за производителност и за self-test.
//
// Разкоментирай следващия ред за демо режим:
// ===========================================================================
//#define DEMO_BOUNDED

#define MERGE_PART 16

class Kangaroo;

#pragma pack(push, 1)
typedef struct {
    int128_t x;
    int128_t d;
    uint32_t kIdx;
    uint32_t h;
} DP;

typedef struct {
    uint32_t header;
    uint32_t nbDP;
    uint32_t threadId;
    uint32_t processId;
    uint32_t gpuId;
} DPHEADER;
#pragma pack(pop)

typedef struct {
    uint32_t nbDP;
    DP* dp;
} DP_CACHE;

typedef struct {
    Kangaroo* obj;
    int  threadId;
    bool isRunning;
    bool hasStarted;
    bool isWaiting;
    uint64_t nbKangaroo;
    Int* px;
    Int* py;
    Int* a;
    uint32_t* state;         // <-- заменя Int* b

    SOCKET clientSock;
    char* clientInfo;
    uint32_t hStart;
    uint32_t hStop;
    char* part1Name;
    char* part2Name;
#ifdef WITHGPU
    int  gridSizeX;
    int  gridSizeY;
    int  gpuId;
#endif
} TH_PARAM;

#define HEADW  0xFA6A8001
#define HEADK  0xFA6A8002
#define HEADKS 0xFA6A8003
#define H_PER_PART (HASH_SIZE / MERGE_PART)

class Kangaroo {
public:
    Kangaroo(Secp256K1* secp, int32_t initDPSize, bool useGpu, std::string& workFile,
        std::string& iWorkFile, uint32_t savePeriod, bool saveKangaroo,
        bool saveKangarooByServer, double maxStep, int wtimeout, int sport,
        int ntimeout, std::string serverIp, std::string outputFile, bool splitWorkfile);

    void Run(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize);
    void RunServer();
    bool ParseConfigFile(std::string& fileName);
    bool LoadWork(std::string& fileName);
    void Check(std::vector<int> gpuId, std::vector<int> gridSize);
    void MergeDir(std::string& dirname, std::string& dest);
    bool MergeWork(std::string& file1, std::string& file2, std::string& dest, bool printStat = true);
    void WorkInfo(std::string& fileName);
    bool MergeWorkPart(std::string& file1, std::string& file2, bool printStat);
    bool MergeWorkPartPart(std::string& part1Name, std::string& part2Name);
    static void CreateEmptyPartWork(std::string& partName);
    void CheckWorkFile(int nbCore, std::string& fileName);
    void CheckPartition(int nbCore, std::string& partName);
    bool FillEmptyPartFromFile(std::string& partName, std::string& fileName, bool printStat);

    void SolveKeyCPU(TH_PARAM* p);
    void SolveKeyGPU(TH_PARAM* p);
    bool HandleRequest(TH_PARAM* p);
    bool MergePartition(TH_PARAM* p);
    bool CheckPartition(TH_PARAM* p);
    bool CheckWorkFile(TH_PARAM* p);
    void ProcessServer();

    void AddConnectedClient();
    void RemoveConnectedClient();
    void RemoveConnectedKangaroo(uint64_t nb);

    bool IsDP(uint64_t x);
    void SetDP(int size);

    // --- Ро ---
    void CreateHerd(int nbKangaroo, Int* px, Int* py, Int* a, uint32_t* state,
        int firstType, bool lock = true);
    void CreateJumpTable();
    void InitGLV();
    void CanonicalizeHost(Point* P, uint32_t* outE, uint32_t* outNeg);
    void UnpackScalar(Int* A, uint32_t st, Int* aOut);
    bool SolveCollision(COLLISION& c);
    bool SelfTest();

    bool AddToTable(uint64_t h, int128_t* x, int128_t* d);

    void ComputeExpected(double dp, double* op, double* ram, double* overHead = NULL);
    void InitRange();
    void InitSearchKey();
    std::string GetTimeStr(double s);
    bool Output(Int* pk, char sInfo, int sType);
    bool Output(Int* pk, uint32_t tIdx);

    void SaveWork(std::string fileName, FILE* f, int type, uint64_t totalCount, double totalTime);
    void SaveWork(uint64_t totalCount, double totalTime, TH_PARAM* threads, int nbThread);
    void SaveServerWork();
    void FetchWalks(uint64_t nbWalk, Int* x, Int* y, Int* a, Int* b);
    void FetchWalks(uint64_t nbWalk, std::vector<int128_t>& kangs, Int* x, Int* y, Int* a, Int* b);
    void FectchKangaroos(TH_PARAM* threads);
    FILE* ReadHeader(std::string fileName, uint32_t* version, int type);
    bool SaveHeader(std::string fileName, FILE* f, int type, uint64_t totalCount, double totalTime);
    int FSeek(FILE* stream, uint64_t pos);
    uint64_t FTell(FILE* stream);
    int IsDir(std::string dirName);
    bool IsEmpty(std::string fileName);
    static std::string GetPartName(std::string& partName, int i, bool tmpPart);
    static FILE* OpenPart(std::string& partName, char* mode, int i, bool tmpPart = false);
    uint32_t CheckHash(uint32_t h, uint32_t nbItem, HashTable* hT, FILE* f);

    void AcceptConnections(SOCKET server_soc);
    int WaitFor(SOCKET sock, int timeout, int mode);
    int Write(SOCKET sock, char* buf, int bufsize, int timeout);
    int Read(SOCKET sock, char* buf, int bufsize, int timeout);
    bool GetConfigFromServer();
    bool ConnectToServer(SOCKET* retSock);
    void InitSocket();
    void WaitForServer();
    int32_t GetServerStatus();
    bool SendKangaroosToServer(std::string& fileName, std::vector<int128_t>& kangs);
    bool GetKangaroosFromServer(std::string& fileName, std::vector<int128_t>& kangs);
    bool SendToServer(std::vector<ITEM>& dp, uint32_t threadId, uint32_t gpuId);

#ifdef WIN64
    HANDLE ghMutex;
    HANDLE saveMutex;
    THREAD_HANDLE LaunchThread(LPTHREAD_START_ROUTINE func, TH_PARAM* p);
#else
    pthread_mutex_t  ghMutex;
    pthread_mutex_t  saveMutex;
    THREAD_HANDLE LaunchThread(void* (*func) (void*), TH_PARAM* p);
#endif

    void JoinThreads(THREAD_HANDLE* handles, int nbThread);
    void FreeHandles(THREAD_HANDLE* handles, int nbThread);
    void Process(TH_PARAM* params, std::string unit);

    uint64_t getCPUCount();
    uint64_t getGPUCount();
    bool isAlive(TH_PARAM* p);
    bool hasStarted(TH_PARAM* p);
    bool isWaiting(TH_PARAM* p);

    Secp256K1* secp;
    HashTable hashTable;          // не се използва в Ро режим, но Thread.cpp го реферира
    uint64_t counters[256];
    int  nbCPUThread;
    int  nbGPUThread;
    double startTime;

    int rangePower;
    Int rangeStart;
    Int rangeEnd;
    Int rangeWidth;
    Int rangeWidthDiv2;
    Int rangeWidthDiv4;
    Int rangeWidthDiv8;

    uint64_t dMask;
    uint32_t dpSize;
    int32_t initDPSize;
    uint64_t collisionInSameHerd;

    std::vector<Point> keysToSearch;
    std::unordered_set<uint32_t> foundTargets;   // по ИНДЕКС на таргета

    Point keyToSearch;
    Point keyToSearchNeg;
    uint32_t keyIdx;
    bool endOfSearch;
    bool useGpu;
    double expectedNbOp;
    double expectedMem;
    double maxStep;
    uint64_t totalRW;

    Int jumpA[NB_JUMP];
    Int jumpB[NB_JUMP];
    Int jumpPointx[NB_JUMP];
    Int jumpPointy[NB_JUMP];
    Int jCGtable[3 * NB_JUMP];    // lambda^-e * c_i mod n

    // GLV / полеви константи
    Int LAMBDA, LAMBDA2;
    Int BETA_H, BETA2_H, PHALF_H;

    int CPU_GRP_SIZE;

    std::string outputFile;
    FILE* fRead;
    uint64_t offsetCount;
    double offsetTime;
    int64_t nbLoadedWalk;
    std::string workFile;
    std::string inputFile;
    int  saveWorkPeriod;
    bool saveRequest;
    bool saveKangaroo;
    bool saveKangarooByServer;
    int wtimeout;
    int ntimeout;
    bool splitWorkfile;

    int port;
    std::string lastError;
    std::string serverIp;
    char* hostInfo;
    int   hostInfoLength;
    int   hostAddrType;
    bool  clientMode;
    bool  isConnected;
    SOCKET serverConn;
    std::vector<DP_CACHE> recvDP;
    std::vector<DP_CACHE> localCache;
    std::string serverStatus;
    int connectedClient;
    uint32_t pid;
};
#endif // KANGAROO_H