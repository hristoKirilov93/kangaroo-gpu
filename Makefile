#---------------------------------------------------------------------
# Kangaroo (JLP fork + RCAsm cubin) — Linux Makefile
#
# Build:
#   make gpu=1 CCAP=89        # RTX 40xx (Ada, напр. 4090/4080)
#   make gpu=1 CCAP=120       # RTX 50xx (Blackwell, 5090/5080)
#
# Ако CUDA не е в /usr/local/cuda:
#   make gpu=1 CCAP=89 CUDA=/usr/local/cuda-12.4
#
# CallCubin.cpp е добавен (на Windows влизаше през VS проекта).
# Линкваме -lcuda (driver API, за cuModuleLoad на cubin-а).
#---------------------------------------------------------------------

.DEFAULT_GOAL := all

OBJDIR = obj

SRCCOMMON = SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
      Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

ifdef gpu
OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      GPU/GPUEngine.o Kangaroo.o HashTable.o Thread.o \
      Check.o Backup.o Network.o Merge.o PartMerge.o CallCubin.o)
else
OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      Kangaroo.o HashTable.o Thread.o Check.o Backup.o \
      Network.o Merge.o PartMerge.o)
endif

CXX      = g++
CUDA    ?= /usr/local/cuda
CXXCUDA ?= /usr/bin/g++
NVCC     = $(CUDA)/bin/nvcc
CCAP    ?= 89

ifdef gpu
ifdef debug
CXXFLAGS = -DWITHGPU -m64 -mssse3 -Wno-unused-result -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS = -DWITHGPU -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif
LFLAGS   = -lpthread -L$(CUDA)/lib64 -lcudart -lcuda
else
CXXFLAGS = -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I$(CUDA)/include
LFLAGS   = -lpthread
endif

#--------------------------------------------------------------------

ifdef gpu
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC \
	  -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include \
	  -gencode=arch=compute_$(CCAP),code=sm_$(CCAP) \
	  -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: kangaroo

kangaroo: $(OBJET)
	@echo Making Kangaroo...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

$(OBJET): | $(OBJDIR) $(OBJDIR)/SECPK1 $(OBJDIR)/GPU

$(OBJDIR):
	mkdir -p $(OBJDIR)
$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU
$(OBJDIR)/SECPK1: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p SECPK1

clean:
	@echo Cleaning...
	@rm -f obj/*.o obj/GPU/*.o obj/SECPK1/*.o
