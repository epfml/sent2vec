/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_SHMEM_MATRIX_H
#define FASTTEXT_SHMEM_MATRIX_H

#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>

#include "matrix.h"
#include "real.h"

namespace fasttext {

class ShmemMatrix : public Matrix {
  public:
    ShmemMatrix(const char*, const int64_t, const int64_t);
    ~ShmemMatrix();

    Matrix& operator=(const Matrix&) = delete;
    void save(std::ostream&) = delete;
    void load(std::istream&) = delete;

    static std::shared_ptr<ShmemMatrix> load(std::istream&, const char*);
};

}

#endif
