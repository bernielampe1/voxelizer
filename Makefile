# make            # no OpenMP
# make USE_OMP=1  # try OpenMP (GCC/Clang with libomp/libgomp)

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter

ifeq ($(USE_OMP),1)
  CXXFLAGS += -fopenmp
endif

BIN := voxelize
SRC := $(BIN).cpp

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	@rm -f voxelize
