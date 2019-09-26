#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x -fPIC -m64
OBJS = args.o dictionary.o productquantizer.o matrix.o shmem_matrix.o qmatrix.o vector.o model.o utils.o fasttext.o
INCLUDES = -I.
ifneq ($(shell uname),Darwin)
	LINK_RT := -lrt
endif

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: src/args.cc src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

productquantizer.o: src/productquantizer.cc src/productquantizer.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/productquantizer.cc

matrix.o: src/matrix.cc src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

shmem_matrix.o: src/shmem_matrix.cc src/shmem_matrix.h
	$(CXX) $(CXXFLAGS) -c src/shmem_matrix.cc

qmatrix.o: src/qmatrix.cc src/qmatrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/qmatrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

fasttext.o: src/fasttext.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/fasttext.cc

fasttext: $(OBJS) src/fasttext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o fasttext $(LINK_RT)

sent2vec.o: $(OBJS) src/sent2vec.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/sent2vec.cc

so: $(OBJS) sent2vec.o
	$(CXX) $(CXXFLAGS) $(OBJS) sent2vec.o -shared -o libsent2vec.so $(LINK_RT)

clean:
	rm -rf *.o *.so fasttext
