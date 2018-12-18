/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <errno.h>
#include <fcntl.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "shmem_matrix.h"

namespace fasttext {

ShmemMatrix::ShmemMatrix(const char* name, const int64_t m, const int64_t n) {
  m_ = m;
  n_ = n;

  // Open an existing shared memory segment
  int fd = shm_open(name, O_RDONLY, 0644);
  if (fd == -1) {
    perror("ShmemMatrix::ShmemMatrix: shm_open failed");
    exit(-1);
  }

  // Map the shared memory segment
  size_t size = m_ * n_ * sizeof(real);
  void* ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (ptr == (void*)-1) {
    perror("ShmemMatrix::ShmemMatrix: mmap failed");
    exit(-1);
  } else {
    data_ = (real*)ptr;
  }

  // Close the file descriptor
  int ret = close(fd);
  if (ret == -1) {
    perror("ShmemMatrix::ShmemMatrix: close failed");
    exit(-1);
  }
}

ShmemMatrix::~ShmemMatrix() {
  // Unmap the shared memory segment
  size_t size = m_ * n_ * sizeof(real);
  int ret = munmap((void*)data_, size);
  if (ret == -1) {
    perror("ShmemMatrix::~ShmemMatrix: munmap failed");
    exit(-1);
  }
  data_ = nullptr;
}

std::shared_ptr<ShmemMatrix> ShmemMatrix::load(std::istream& in, const char* name) {
  int64_t m, n;
  in.read((char*)&m, sizeof(int64_t));
  in.read((char*)&n, sizeof(int64_t));
  size_t size = m * n * sizeof(real);

  // Create a shared memory segment
  bool new_segment = true;
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0644);
  if (fd == -1) {
    if (errno == EEXIST) {
      new_segment = false;
    } else {
      perror("ShmemMatrix::load: shm_open failed");
      exit(-1);
    }
  }

  if (new_segment) {
    // Set the size for shared memory segment
    int ret = ftruncate(fd, size);
    if (ret == -1) {
      perror("ShmemMatrix::load: ftruncate failed");
      exit(-1);
    }

    // Map the shared memory segment
    void* ptr = mmap(NULL, size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == (void*)-1) {
      perror("ShmemMatrix::load: mmap failed");
      exit(-1);
    }

    // Populate the shared memory segment
    in.read((char*)ptr, size);

    // Unmap the shared memory segment
    ret = munmap(ptr, size);
    if (ret == -1) {
      perror("ShmemMatrix::load: munmap failed");
      exit(-1);
    }
  } else {
    // Seek in the stream to skip the input matrix data
    in.seekg(size, in.cur);
  }

  auto matrix = std::make_shared<ShmemMatrix>(name, m, n);

  //TODO: Unlink is only safe to do once every process has opened the shared
  //      memory segment. This is hard to do from the code. One way would be
  //      to put a refcount in the segment itself, which would also require
  //      the segment to be opened as O_RDWR, and the refcount should be
  //      protected by an IPC semaphore.
  //// Unlink the shared memory segment
  //int ret = shm_unlink(name);
  //if (ret == -1) {
  //  perror("shm_unlink failed");
  //  exit(-1);
  //}

  return matrix;
}

}
