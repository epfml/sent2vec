/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"
#include "shmem_matrix.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <stdio.h>


namespace fasttext {

FastText::FastText() : quant_(false) {}

void FastText::getVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getNgrams(word);
  vec.zero();
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
    vec.addRow(*input_, *it);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::saveVectors() {
  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  int32_t version;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version != FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void FastText::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  std::ofstream ofs(fn, std::ofstream::binary);
  if (!ofs.is_open()) {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  signModel(ofs);
  args_->save(ofs);
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  if (quant_) {
    qinput_->save(ofs);
  } else {
    input_->save(ofs);
  }

  ofs.write((char*)&(args_->qout), sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->save(ofs);
  } else {
    output_->save(ofs);
  }

  ofs.close();
}

void FastText::loadModel(const std::string& filename,
                         const bool predict_mode /* = false */) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!checkModel(ifs)) {
    std::cerr << "Model file has wrong file format!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (predict_mode) {
    loadModelForPredict(ifs);
  } else {
    loadModel(ifs);
  }
  ifs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  qinput_ = std::make_shared<QMatrix>();
  qoutput_ = std::make_shared<QMatrix>();
  args_->load(in);

  dict_->load(in);

  bool quant_input;
  in.read((char*) &quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    qinput_->load(in);
  } else {
    input_->load(in);
  }

  in.read((char*) &args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->load(in);
  } else {
    output_->load(in);
  }

  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);

  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::loadModelForPredict(std::istream& in) {
  args_ = std::make_shared<Args>();
  args_->load(in);

  dict_ = std::make_shared<Dictionary>(args_);
  dict_->load(in);

  in.read((char*) &quant_, sizeof(bool));

  input_ = ShmemMatrix::load(in, "s2v_input_matrix");

  in.read((char*) &args_->qout, sizeof(bool));

  output_ = std::make_shared<Matrix>();
  in.read((char*) &(output_->m_), sizeof(int64_t));
  in.read((char*) &(output_->n_), sizeof(int64_t));

  model_ = std::make_shared<Model>(input_, output_, args_, 0);

  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::printInfo(real progress, real loss) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  real lr = args_->lr * (1.0 - progress);
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cerr << std::fixed;
  std::cerr << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cerr << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cerr << "  lr: " << std::setprecision(6) << lr;
  std::cerr << "  loss: " << std::setprecision(6) << loss;
  std::cerr << "  eta: " << etah << "h" << etam << "m ";
  std::cerr << std::flush;
}

std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
  Vector norms(input_->m_);
  input_->l2NormRow(norms);
  std::vector<int32_t> idx(input_->m_, 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(),
      [&norms, eosid] (size_t i1, size_t i2) {
      return eosid ==i1 || (eosid != i2 && norms[i1] > norms[i2]);
      });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
}

void FastText::quantize(std::shared_ptr<Args> qargs) {
  if (qargs->output.empty()) {
      std::cerr<<"No model provided!"<<std::endl; exit(1);
  }
  loadModel(qargs->output + ".bin");

  args_->input = qargs->input;
  args_->qout = qargs->qout;
  args_->output = qargs->output;


  if (qargs->cutoff > 0 && qargs->cutoff < input_->m_) {
    auto idx = selectEmbeddings(qargs->cutoff);
    dict_->prune(idx);
    std::shared_ptr<Matrix> ninput =
      std::make_shared<Matrix> (idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i,j) = input_->at(idx[i], j);
      }
    }
    input_ = ninput;
    if (qargs->retrain) {
      args_->epoch = qargs->epoch;
      args_->lr = qargs->lr;
      args_->thread = qargs->thread;
      args_->verbose = qargs->verbose;
      tokenCount = 0;
      std::vector<std::thread> threads;
      for (int32_t i = 0; i < args_->thread; i++) {
        threads.push_back(std::thread([=]() { trainThread(i); }));
      }
      for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
      }
    }
  }

  qinput_ = std::make_shared<QMatrix>(*input_, qargs->dsub, qargs->qnorm);

  if (args_->qout) {
    qoutput_ = std::make_shared<QMatrix>(*output_, 2, qargs->qnorm);
  }

  quant_ = true;
  saveModel();
}

void FastText::supervised(Model& model, real lr,
                          const std::vector<int32_t>& line,
                          const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::cbow(Model& model, real lr,
                    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::sent2vec(Model& model, real lr, const std::vector<int32_t>& line){
  if (line.size() <= 1) return;
  std::vector<int32_t> context;
  std::uniform_real_distribution<> uniform(0, 1);
  for (int32_t i=0; i<line.size(); ++i){
    if (uniform(model.rng) > dict_->getPDiscard(line[i]) || dict_->getTokenCount(line[i]) < args_->minCountLabel)
      continue;
    context = line;
    context[i] = 0;
    dict_->addNgrams(context, args_->wordNgrams, args_->dropoutK, model.rng);
    model.update(context, line[i], lr);
  }
}

void FastText::test(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << "N" << "\t" << nexamples << std::endl;
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << "\t" << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << "\t" << precision / nlabels << std::endl;
  std::cerr << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(std::istream& in, int32_t k,
                       std::vector<std::pair<real,std::string>>& predictions) const {
  std::vector<int32_t> words, labels;
  dict_->getLine(in, words, labels, model_->rng);
  if (words.empty()) return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real,int32_t>> modelPredictions;
  model_->predict(words, k, modelPredictions, hidden, output);
  predictions.clear();
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
    predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
  std::vector<std::pair<real,std::string>> predictions;
  while (in.peek() != EOF) {
    predict(in, k, predictions);
    if (predictions.empty()) {
      std::cout << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << " ";
      }
      std::cout << it->second;
      if (print_prob) {
        std::cout << " " << exp(it->first);
      }
    }
    std::cout << std::endl;
  }
}

void FastText::wordVectors() {
  std::string word;
  Vector vec(args_->dim);
  while (std::cin >> word) {
    getVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
}

void FastText::sentenceVectors() {
  Vector vec(args_->dim);
  std::string sentence;
  Vector svec(args_->dim);
  std::string word;
  while (std::getline(std::cin, sentence)) {
    std::istringstream iss(sentence);
    svec.zero();
    int32_t count = 0;
    while(iss >> word) {
      getVector(vec, word);
      vec.mul(1.0 / vec.norm());
      svec.addVector(vec);
      count++;
    }
    svec.mul(1.0 / count);
    std::cout << sentence << " " << svec << std::endl;
  }
}

void FastText::ngramVectors(std::string word) {
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  Vector vec(args_->dim);
  dict_->getNgrams(word, ngrams, substrings);
  for (int32_t i = 0; i < ngrams.size(); i++) {
    vec.zero();
    if (ngrams[i] >= 0) {
      vec.addRow(*input_, ngrams[i]);
    }
    std::cout << substrings[i] << " " << vec << std::endl;
  }
}

void FastText::textVectors() {
  std::vector<int32_t> line, labels;
  Vector vec(args_->dim);
  while (std::cin.peek() != EOF) {
    dict_->getLine(std::cin, line, labels, model_->rng);
    vec.zero();
    if (args_->model == model_name::sent2vec){
      dict_->addNgrams(line, args_->wordNgrams);
    }
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      vec.addRow(*input_, *it);
    }
    if (!line.empty()) {
      vec.mul(1.0 / line.size());
    }
    std::cout << vec << std::endl;
  }
}

void FastText::textVectorThread(int thread_id, std::shared_ptr<std::vector<std::string>> sentences, std::shared_ptr<Matrix> emb, int num_threads) {
  std::vector<int32_t> line, labels;
  for (int sent_idx=thread_id; sent_idx < sentences->size(); sent_idx+=num_threads) {
    Vector vec(args_->dim);
    textVector(sentences->operator[](sent_idx), vec, line, labels);
    emb->addRow(vec, sent_idx, 1.);
  }
}

void FastText::textVectors(std::vector<std::string>& sentences, int num_threads, std::vector<real>& final) {
  std::shared_ptr<Matrix> emb;
  std::shared_ptr<std::vector<std::string>> sents;
  sents = std::make_shared<std::vector<std::string>>(sentences);
  emb = std::make_shared<Matrix>(sentences.size(), args_->dim);
  emb->zero();
  std::vector<std::thread> threads;
  for (int32_t i = 0; i < num_threads; i++) {
    threads.push_back(std::thread([=]() { textVectorThread(i, sents, emb, num_threads); }));
  }
  for (auto it = threads.begin(); it != threads.end(); ++it) {
    it->join();
  }
  memcpy(&final[0], &emb->data_[0], emb->m_*emb->n_ * sizeof(real));
}

void FastText::textVector(std::string text, Vector& vec, std::vector<int32_t>& line, std::vector<int32_t>& labels) {
  std::istringstream text_stream(text);
  dict_->getLine(text_stream, line, labels, model_->rng);
  vec.zero();
  if (args_->model == model_name::sent2vec){
    dict_->addNgrams(line, args_->wordNgrams);
  }
  for (auto it = line.cbegin(); it != line.cend(); ++it) {
    vec.addRow(*input_, *it);
  }
  if (!line.empty()) {
    vec.mul(1.0 / line.size());
  }
}

void FastText::printWordVectors() {
  wordVectors();
}

void FastText::printSentenceVectors() {
  if (args_->model == model_name::sup || args_->model == model_name::sent2vec) {
    textVectors();
  } else {
    sentenceVectors();
  }
}

void FastText::precomputeWordVectors(Matrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  std::cerr << "Pre-computing word vectors...";
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getVector(vec, word);
    real norm = vec.norm();
    wordVectors.addRow(vec, i, 1.0 / norm);
  }
  std::cerr << " done." << std::endl;
}

void FastText::precomputeSentenceVectors(Matrix& sentenceVectors,std::ifstream& in) {
  Vector vec(args_->dim);
  sentenceVectors.zero();
  std::cerr << "Pre-computing sentence vectors...";
  std::vector<int32_t> line;
  std::vector<int32_t> labels;
  int32_t i = 0;
  while (i < sentenceVectors.m_) {
  
    dict_->getLine(in, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);

    vec.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      vec.addRow(*input_, *it);
    }
    if (!line.empty()) {
      vec.mul(1.0 / line.size());
    }
    real norm = vec.norm();
    if(norm != 0)
        sentenceVectors.addRow(vec, i, 1.0 / norm);
    i++;
  }
  std::cerr << " done." << std::endl;
}

void FastText::findNN(const Matrix& wordVectors, const Vector& queryVec,
                      int32_t k, const std::set<std::string>& banSet) {
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }
  std::priority_queue<std::pair<real, std::string>> heap;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    real dp = wordVectors.dotRow(queryVec, i);
    heap.push(std::make_pair(dp / queryNorm, word));
  }
  int32_t i = 0;
  while (i < k && heap.size() > 0) {
    auto it = banSet.find(heap.top().second);
    if (it == banSet.end()) {
      std::cout << heap.top().second << " " << heap.top().first << std::endl;
      i++;
    }
    heap.pop();
  }
}

void FastText::findNNSent(const Matrix& sentenceVectors, const Vector& queryVec,
                          int32_t k, const std::set<std::string>& banSet, int64_t numSent, 
                          const std::vector<std::string>& sentences) {
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }
  std::priority_queue<std::pair<real, std::string>> heap;
  Vector vec(args_->dim);

  for (int32_t i = 0; i < numSent; i++) {
    std::string sentence = std::to_string(i) + " " + sentences[i];
    real dp = sentenceVectors.dotRow(queryVec, i);
    heap.push(std::make_pair(dp / queryNorm, sentence));
  }

  int32_t i = 0;
  while (i < k && heap.size() > 0) {
    auto it = banSet.find(heap.top().second);
    if (!std::isnan(heap.top().first)) {
      std::cout << heap.top().first << " " 
                                    << heap.top().second << " " 
                                    << std::endl;
      i++;
    }
    heap.pop();
  }
}


void FastText::nn(int32_t k) {
  std::string queryWord;
  Vector queryVec(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cerr << "Query word? " << std::endl;
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    getVector(queryVec, queryWord);
    findNN(wordVectors, queryVec, k, banSet);
    std::cerr << "Query word? " << std::endl;
  }
}

void FastText::analogies(int32_t k) {
  std::string word;
  Vector buffer(args_->dim), query(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cerr << "Query triplet (A - B + C)? " << std::endl;
  while (true) {
    banSet.clear();
    query.zero();
    std::cin >> word;
    banSet.insert(word);
    getVector(buffer, word);
    query.addVector(buffer, 1.0);
    std::cin >> word;
    banSet.insert(word);
    getVector(buffer, word);
    query.addVector(buffer, -1.0);
    std::cin >> word;
    banSet.insert(word);
    getVector(buffer, word);
    query.addVector(buffer, 1.0);

    findNN(wordVectors, query, k, banSet);
    std::cerr << "Query triplet (A - B + C)? " << std::endl;
  }
}

void FastText::nnSent(int32_t k, std::string filename) {  
  std::string sentence;
  std::ifstream in1(filename);
  int64_t n = 0;

  Vector buffer(args_->dim), query(args_->dim);
  std::vector<std::string> sentences;

  std::vector<int32_t> line, labels;
  std::ifstream in2(filename);

  while (in2.peek() != EOF) {
    std::getline(in2, sentence);
    sentences.push_back(sentence);
    n++;
  }
  std::cout << "Number of sentences in the corpus file is " << n << "." << std::endl ;
  Matrix sentenceVectors(n+1, args_->dim);

  precomputeSentenceVectors(sentenceVectors, in1);
  std::set<std::string> banSet;

  std::cerr << "Query sentence? " << std::endl;
  while (std::cin.peek() != EOF) {
    query.zero();
    dict_->getLine(std::cin, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    buffer.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      buffer.addRow(*input_, *it);
    }
    if (!line.empty()) {
      buffer.mul(1.0 / line.size());
    }
    query.addVector(buffer, 1.0);

    findNNSent(sentenceVectors, query, k, banSet, n, sentences);
    std::cout << std::endl;
    std::cerr << "Query sentence? " << std::endl;
  }
}


void FastText::analogiesSent(int32_t k, std::string filename) {
  std::string sentence;
  std::ifstream in1(filename);
  int64_t n = 0;   
  
  Vector buffer(args_->dim), query(args_->dim);
  std::vector<std::string> sentences;
  
  std::vector<int32_t> line, labels;

  std::ifstream in2(filename);
  
  while (in2.peek() != EOF) {
    std::getline(in2, sentence);
    sentences.push_back(sentence);
    n++;
  }
  std::cout << "Number of sentences in the corpus file is " << n << "." << std::endl ;

  Matrix sentenceVectors(n+1, args_->dim);

  precomputeSentenceVectors(sentenceVectors, in1);
  std::set<std::string> banSet;
  std::cerr << "Query triplet sentences (A - B + C)? " << std::endl;
  while (true) {
    banSet.clear();
    query.zero();
    dict_->getLine(std::cin, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    buffer.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      buffer.addRow(*input_, *it);
    }
    if (!line.empty()) {
      buffer.mul(1.0 / line.size());
    }
    query.addVector(buffer, 1.0);

    dict_->getLine(std::cin, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    buffer.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      buffer.addRow(*input_, *it);
    }
    if (!line.empty()) {
      buffer.mul(1.0 / line.size());
    }

    query.addVector(buffer, -1.0);

    dict_->getLine(std::cin, line, labels, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    buffer.zero();
    for (auto it = line.cbegin(); it != line.cend(); ++it) {
      buffer.addRow(*input_, *it);
    }
    if (!line.empty()) {
      buffer.mul(1.0 / line.size());
    }

    query.addVector(buffer, 1.0);

    findNNSent(sentenceVectors, query, k, banSet, n, sentences);
    std::cerr << "Query triplet sentences (A - B + C)? " << std::endl;
  }
}


void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, threadId);
  if (args_->model == model_name::sup) {
    model.setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model.setTargetCounts(dict_->getCounts(entry_type::word));
  }

  const int64_t ntokens = dict_->ntokens();
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  while (tokenCount < args_->epoch * ntokens) {
    real progress = real(tokenCount) / (args_->epoch * ntokens);
    real lr = args_->lr * (1.0 - progress);
    localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
    if (args_->model == model_name::sup) {
      supervised(model, lr, line, labels);
    } else if (args_->model == model_name::sent2vec) {
      sent2vec(model, lr, line);
    } else if (args_->model == model_name::cbow) {
      cbow(model, lr, line);
    } else if (args_->model == model_name::sg) {
      skipgram(model, lr, line);
    }
    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount += localTokenCount;
      localTokenCount = 0;
      if (threadId == 0 && args_->verbose > 1) {
        printInfo(progress, model.getLoss());
      }
    }
  }
  if (threadId == 0 && args_->verbose > 0) {
    printInfo(1.0, model.getLoss());
    std::cerr << std::endl;
  }
  ifs.close();
}

void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    std::cerr << "Dimension of pretrained vectors does not match -dim option"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  dict_->threshold(1, 0);
  input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->data_[idx * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args) {
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    std::cerr << "Cannot use stdin for training!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  dict_->readFromFile(ifs);
  ifs.close();

  if (args_->pretrainedVectors.size() != 0) {
    loadVectors(args_->pretrainedVectors);
  } else {
    input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
    input_->uniform(1.0 / args_->dim);
  }

  if (args_->model == model_name::sup) {
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  } else {
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();

  start = clock();
  tokenCount = 0;
  if (args_->thread > 1) {
    std::vector<std::thread> threads;
    for (int32_t i = 0; i < args_->thread; i++) {
      threads.push_back(std::thread([=]() { trainThread(i); }));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
      it->join();
    }
  } else {
    trainThread(0);
  }
  model_ = std::make_shared<Model>(input_, output_, args_, 0);

  saveModel();
  if (args_->model != model_name::sup && args_->model != model_name::sent2vec) {
    saveVectors();
    if (args_->saveOutput > 0) {
      saveOutput();
    }
  }
}

int FastText::getDimension() const {
    return args_->dim;
}

}
