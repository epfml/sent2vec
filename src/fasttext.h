/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_FASTTEXT_H
#define FASTTEXT_FASTTEXT_H

#define FASTTEXT_VERSION 11 /* Version 1a */
#define FASTTEXT_FILEFORMAT_MAGIC_INT32 793712314

#include <time.h>

#include <atomic>
#include <memory>
#include <set>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "qmatrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
  private:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;

    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;

    std::shared_ptr<QMatrix> qinput_;
    std::shared_ptr<QMatrix> qoutput_;

    std::shared_ptr<Model> model_;

    std::atomic<int64_t> tokenCount;
    clock_t start;
    void signModel(std::ostream&);
    bool checkModel(std::istream&);

    bool quant_;

  public:
    FastText();

    void getVector(Vector&, const std::string&) const;
    void getVector(Vector&, int32_t) const;
    void saveVectors();
    void saveOutput();
    void saveModel();
    void saveModel(int32_t);
    void saveDict();

    void loadModel(const std::string&, const bool inference_mode = false, const bool shared_mem_enabled = true,
                   const int timeout_sec = -1);
    void loadModel(std::istream&, bool load_output_matrix = true);
    void loadModelWithSharedMemory(std::istream&, const std::string&, const int);
    void loadDict(const std::string&);
    void loadDict(std::istream&);
    void printInfo(real, real);

    void supervised(Model&, real, const std::vector<int32_t>&,
                    const std::vector<int32_t>&);
    void cbow(Model&, real, const std::vector<int32_t>&);
    void sent2vec(Model&, real, const std::vector<int32_t>&);
    void cbowCWNgrams(Model&, real, std::vector<int32_t>&);
    void skipgram(Model&, real, const std::vector<int32_t>&);
    std::vector<int32_t> selectEmbeddings(int32_t) const;
    void quantize(std::shared_ptr<Args>);
    void test(std::istream&, int32_t);
    void predict(std::istream&, int32_t, bool);
    void predict(std::istream&, int32_t, std::vector<std::pair<real,std::string>>&) const;
    void wordVectors();
    void sentenceVectors();
    void ngramVectors(std::string);
    void textVectors();
    void textVectorThread(int, std::shared_ptr<std::vector<std::string>>, std::shared_ptr<Matrix>, int);
    void textVectors(std::vector<std::string>&, int, std::vector<real>&);
    void textVector(std::string, Vector&, std::vector<int32_t>&, std::vector<int32_t>&);
    void printWordVectors();
    void printVocabularyVectors(bool);
    void printSentenceVectors();
    std::vector<std::string> getVocab();
    std::vector<int64_t> getUnigramsCounts();
    void trainThread(int32_t);

    void savedDictTrain(std::shared_ptr<Args>);
    void trainDict(std::shared_ptr<Args>);


    void train(std::shared_ptr<Args>);
    void precomputeWordVectors(Matrix&);
    void precomputeSentenceVectors(Matrix&,std::ifstream&);
    void findNN(const Matrix&, const Vector&, int32_t,
                const std::set<std::string>&);
    void findNNSent(const Matrix&, const Vector&, int32_t,
                const std::set<std::string>&, int64_t, const std::vector<std::string>&);
    void nn(int32_t);
    void analogies(int32_t);
    void nnSent(int32_t, std::string );
    void analogiesSent(int32_t, std::string );

    void loadVectors(std::string);
    int getDimension() const;
};

}

#endif
