/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>

#include "fasttext.h"
#include "args.h"

using namespace fasttext;

void printUsage() {
  std::cerr
    << "usage: fasttext <command> <args>\n\n"
    << "The commands supported by fasttext are:\n\n"
    << "  supervised              train a supervised classifier\n"
    << "  sent2vec                train unsupervised sentence embeddings\n"
    << "  quantize                quantize a model to reduce the memory usage\n"
    << "  test                    evaluate a supervised classifier\n"
    << "  predict                 predict most likely labels\n"
    << "  predict-prob            predict most likely labels with probabilities\n"
    << "  skipgram                train a skipgram model\n"
    << "  cbow                    train a cbow model\n"
    << "  print-word-vectors      print word vectors given a trained model\n"
    << "  print-sentence-vectors  print sentence vectors given a trained model\n"
    << "  nn                      query for nearest neighbors\n"
    << "  nnSent                  query for nearest neighbors for sentences\n"
    << "  analogies               query for analogies\n"
    << "  analogiesSent           query for analogies for Sentences\n"
    << std::endl;  
}

void printQuantizeUsage() {
  std::cerr
    << "usage: fasttext quantize <args>"
    << std::endl;
}

void printTestUsage() {
  std::cerr
    << "usage: fasttext test <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPredictUsage() {
  std::cerr
    << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPrintWordVectorsUsage() {
  std::cerr
    << "usage: fasttext print-word-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintSentenceVectorsUsage() {
  std::cerr
    << "usage: fasttext print-sentence-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void printPrintNgramsUsage() {
  std::cerr
    << "usage: fasttext print-ngrams <model> <word>\n\n"
    << "  <model>      model filename\n"
    << "  <word>       word to print\n"
    << std::endl;
}

void quantize(int argc, char** argv) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  if (argc < 3) {
    printQuantizeUsage();
    a->printHelp();
    exit(EXIT_FAILURE);
  }
  a->parseArgs(argc, argv);
  FastText fasttext;
  fasttext.quantize(a);
  exit(0);
}

void printNNUsage() {
  std::cout
    << "usage: fasttext nn <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void printNNSentUsage() {
  std::cerr
    << "usage: fasttext nnSent <model> <corpus> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <corpus>     corpus filename \n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
    std::cout<<"NOTE : A corpus file is required to find similar sentences."<<std::endl;
}

void printAnalogiesUsage() {
  std::cout
    << "usage: fasttext analogies <model> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
}

void printAnalogiesSentUsage() {
  std::cout
    << "usage: fasttext analogiesSent <model> <corpus> <k>\n\n"
    << "  <model>      model filename\n"
    << "  <corpus>     corpus filename \n"
    << "  <k>          (optional; 10 by default) predict top k labels\n"
    << std::endl;
  std::cout<<"NOTE : A corpus file is required to find similar sentences."<<std::endl;
}

void test(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (argc >= 5) {
    k = atoi(argv[4]);
  }

  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));

  std::string infile(argv[3]);
  if (infile == "-") {
    fasttext.test(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.test(ifs, k);
    ifs.close();
  }
  exit(0);
}

void predict(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  int32_t k = 1;
  if (argc >= 5) {
    k = atoi(argv[4]);
  }

  bool print_prob = std::string(argv[1]) == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));

  std::string infile(argv[3]);
  if (infile == "-") {
    fasttext.predict(std::cin, k, print_prob);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.predict(ifs, k, print_prob);
    ifs.close();
  }

  exit(0);
}

void printWordVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintWordVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.printWordVectors();
  exit(0);
}

void printSentenceVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintSentenceVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.printSentenceVectors();
  exit(0);
}

void printNgrams(int argc, char** argv) {
  if (argc != 4) {
    printPrintNgramsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.ngramVectors(std::string(argv[3]));
  exit(0);
}

void nn(int argc, char** argv) {
  int32_t k;
  if (argc == 3) {
    k = 10;
  } else if (argc == 4) {
    k = atoi(argv[3]);
  } else {
    printNNUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.nn(k);
  exit(0);
}

void nnSent(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 10;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printNNSentUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.nnSent(k,std::string(argv[3]));
  exit(0);
}


void analogies(int argc, char** argv) {
  int32_t k;
  if (argc == 3) {
    k = 10;
  } else if (argc == 4) {
    k = atoi(argv[3]);
  } else {
    printAnalogiesUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.analogies(k);
  exit(0);
}

void analogiesSent(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 10;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printAnalogiesSentUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.analogiesSent(k,std::string(argv[3]));
  exit(0);
}

void train(int argc, char** argv) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  a->parseArgs(argc, argv);
  FastText fasttext;
  fasttext.train(a);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(argv[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised" ||
      command == "sent2vec") {
    train(argc, argv);
  } else if (command == "test") {
    test(argc, argv);
  } else if (command == "quantize") {
    quantize(argc, argv);
  } else if (command == "print-word-vectors") {
    printWordVectors(argc, argv);
  } else if (command == "print-sentence-vectors") {
    printSentenceVectors(argc, argv);
  } else if (command == "print-ngrams") {
    printNgrams(argc, argv);
  } else if (command == "nn") {
    nn(argc, argv);
  } else if (command == "nnSent") {
    nnSent(argc, argv);
  } else if (command == "analogies") {
    analogies(argc, argv);
  } else if (command == "analogiesSent") {
    analogiesSent(argc, argv);
  } else if (command == "predict" || command == "predict-prob" ) {
    predict(argc, argv);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
