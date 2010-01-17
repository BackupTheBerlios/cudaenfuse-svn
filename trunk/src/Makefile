################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	 := cuda_enfuse

# Cuda source files (compiled with cudacc)
CUFILES		 := cuda_enfuse.cu \
                cuda_wrappers.cu \
                filter_wrappers.cu
                
# C/C++ source files (compiled with gcc / c++)
CCFILES		 := main.cpp \
                bmploader.cpp

verbose             := 1
nvcc_warn_verbose   := 1

# use -lcuda
#COMMONFLAGS := -lcuda
#OMIT_CUTIL_LIB := 1

################################################################################
# Rules and targets
################################################################################

# Cuda SDK root path
CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_SDK_PATH     := /opt/cuda/sdk

# Auto-configure CUDA paths to search
CUDA_2_0_COMMON    := $(CUDA_SDK_PATH)/common
CUDA_2_3_COMMON    := $(CUDA_SDK_PATH)/C/common
CUDA_COMMON_PATHS  := $(CUDA_2_3_COMMON) \
                      $(CUDA_2_0_COMMON)
CUDA_COMMON        := $(wildcard $(CUDA_COMMON_PATHS))

# Set some CUDA specific variables
ROOTDIR           := $(CUDA_COMMON)
ROOTBINDIR        := bin
ROOTOBJDIR        := obj


# include the CUDA magic makefile
include $(CUDA_COMMON)/common.mk
