## Updates 

Code and pre-trained models related to the [Bi-Sent2vec](https://arxiv.org/abs/1912.12481), cross-lingual extension of Sent2Vec can be found [here](https://github.com/epfml/Bi-sent2vec). 

# Sent2vec

TLDR: This library provides numerical representations (features) for words, short texts, or sentences, which can be used as input to any machine learning task. 

### Table of Contents  

* [Setup and Requirements](#setup-and-requirements)
* [Sentence Embeddings](#sentence-embeddings)
    - [Generating Features from Pre-Trained Models](#generating-features-from-pre-trained-models)
    - [Downloading Sent2vec Pre-Trained Models](#downloading-sent2vec-pre-trained-models)
    - [Train a New Sent2vec Model](#train-a-new-sent2vec-model)
    - [Nearest Neighbour Search and Analogies](#nearest-neighbour-search-and-analogies)
* [Word (Unigram) Embeddings](#unigram-embeddings)
    - [Extracting Word Embeddings from Pre-Trained Models](#extracting-word-embeddings-from-pre-trained-models)
    - [Downloading Pre-Trained Models](#downloading-pre-trained-models)
    - [Train a CBOW Character and Word Ngrams Model](#train-a-cbow-character-and-word-ngrams-model)
* [References](#references)

# Setup and Requirements

Our code builds upon [Facebook's FastText library](https://github.com/facebookresearch/fastText), see also their nice documentation and python interfaces.

To compile the library, simply run a `make` command.

A Cython module allows you to keep the model in memory while inferring sentence embeddings. In order to compile and install the module, run the following from the project root folder:

```
pip install .
```

## Note -  
if you install sent2vec using

```
$ pip install sent2vec
```

then you'll get the wrong package. Please follow the instructions in the README.md to install it correctly.

# Sentence Embeddings

For the purpose of generating sentence representations, we introduce our sent2vec method and provide code and models. Think of it as an unsupervised version of [FastText](https://github.com/facebookresearch/fastText), and an extension of word2vec (CBOW) to sentences. 

The method uses a simple but efficient unsupervised objective to train distributed representations of sentences. The algorithm outperforms the state-of-the-art unsupervised models on most benchmark tasks, and on many tasks even beats supervised models, highlighting the robustness of the produced sentence embeddings, see [*the paper*](https://aclweb.org/anthology/N18-1049) for more details.

## Generating Features from Pre-Trained Models

### Directly from Python

If you've installed the Cython module, you can infer sentence embeddings while keeping the model in memory:

```python
import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('model.bin')
emb = model.embed_sentence("once upon a time .") 
embs = model.embed_sentences(["first sentence .", "another sentence"])
```

Text preprocessing (tokenization and lowercasing) is not handled by the module, check `wikiTokenize.py` for tokenization using NLTK and Stanford NLP. 

An alternative to the Cython module is using the python code provided in the `get_sentence_embeddings_from_pre-trained_models` notebook. It handles tokenization and can be given raw sentences, but does not keep the model in memory. 

#### Running Inference with Multiple Processes

There is an 'inference' mode for loading the model in the Cython module, which loads the model's input matrix into a shared memory segment and doesn't load the output matrix, which is not needed for inference. This is an optimization for the usecase of running inference with multiple independent processes, which would otherwise each need to load a copy of the model into their address space. To use it:
```python
model.load_model('model.bin', inference_mode=True)
```

The model is loaded into a shared memory segment named after the model name. The model will stay in memory until you explicitely remove the shared memory segment. To do it from Python:
```python
model.release_shared_mem('model.bin')
```

### Using the Command-line Interface

Given a pre-trained model `model.bin` (download links see below), here is how to generate the sentence features for an input text. To generate the features, use the `print-sentence-vectors` command and the input text file needs to be provided as one sentence per line:

```
./fasttext print-sentence-vectors model.bin < text.txt
```

This will output sentence vectors (the features for each input sentence) to the standard output, one vector per line.
This can also be used with pipes:

```
cat text.txt | ./fasttext print-sentence-vectors model.bin
```

## Downloading Sent2vec Pre-Trained Models

- [sent2vec_wiki_unigrams](https://drive.google.com/file/d/0B6VhzidiLvjSa19uYWlLUEkzX3c/view?usp=sharing&resourcekey=0-p9iI_hJbCuNiUq5gWz7Qpg) 5GB (600dim, trained on english wikipedia)
- [sent2vec_wiki_bigrams](https://drive.google.com/file/d/0B6VhzidiLvjSaER5YkJUdWdPWU0/view?usp=sharing&resourcekey=0-MVSyokxog2m4EQ4AGsssww) 16GB (700dim, trained on english wikipedia)
- [sent2vec_twitter_unigrams](https://drive.google.com/file/d/0B6VhzidiLvjSaVFLM0xJNk9DTzg/view?usp=sharing&resourcekey=0--yCdYMEuuD2Ml7jIBhJiDw) 13GB (700dim, trained on english tweets)
- [sent2vec_twitter_bigrams](https://drive.google.com/file/d/0B6VhzidiLvjSeHI4cmdQdXpTRHc/view?usp=sharing&resourcekey=0-5wNEK0boM-tRvmkCIb8Txw) 23GB (700dim, trained on english tweets)
- [sent2vec_toronto books_unigrams](https://drive.google.com/file/d/0B6VhzidiLvjSOWdGM0tOX1lUNEk/view?usp=sharing&resourcekey=0-dQDQ3OZWooMbg-g48GRf1Q) 2GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))
- [sent2vec_toronto books_bigrams](https://drive.google.com/file/d/0B6VhzidiLvjSdENLSEhrdWprQ0k/view?usp=sharing&resourcekey=0-c1Qyo6RNF5TRsVzrNXhdRw) 7GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))

(as used in the NAACL2018 paper)

Note: users who downloaded models prior to [this release](https://github.com/epfml/sent2vec/releases/tag/v1) will encounter compatibility issues when trying to use the old models with the latest commit. Those users can still use the code in the release to keep using old models. 

### Tokenizing
Both feature generation as above and also training as below do require that the input texts (sentences) are already tokenized. To tokenize and preprocess text for the above models, you can use

```
python3 tweetTokenize.py <tweets_folder> <dest_folder> <num_process>
```

for tweets, or then the following for wikipedia:
```
python3 wikiTokenize.py corpora > destinationFile
```
Note: For `wikiTokenize.py`, set the `SNLP_TAGGER_JAR` parameter to be the path of `stanford-postagger.jar` which you can download [here](http://www.java2s.com/Code/Jar/s/Downloadstanfordpostaggerjar.htm)

## Train a New Sent2vec Model

To train a new sent2vec model, you first need some large training text file. This file should contain one sentence per line. The provided code does not perform tokenization and lowercasing, you have to preprocess your input data yourself, see above.

You can then train a new model. Here is one example of command:

    ./fasttext sent2vec -input wiki_sentences.txt -output my_model -minCount 8 -dim 700 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000 -maxVocabSize 750000 -numCheckPoints 10

Here is a description of all available arguments:

```
sent2vec -input train.txt -output model

The following arguments are mandatory:
  -input              training file path
  -output             output file path

The following arguments are optional:
  -lr                 learning rate [0.2]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                dimension of word and sentence vectors [100]
  -epoch              number of epochs [5]
  -minCount           minimal number of word occurences [5]
  -minCountLabel      minimal number of label occurences [0]
  -neg                number of negatives sampled [10]
  -wordNgrams         max length of word ngram [2]
  -loss               loss function {ns, hs, softmax} [ns]
  -bucket             number of hash buckets for vocabulary [2000000]
  -thread             number of threads [2]
  -t                  sampling threshold [0.0001]
  -dropoutK           number of ngrams dropped when training a sent2vec model [2]
  -verbose            verbosity level [2]
  -maxVocabSize       vocabulary exceeding this size will be truncated [None]
  -numCheckPoints     number of intermediary checkpoints to save when training [1]
```

## Nearest Neighbour Search and Analogies
Given a pre-trained model `model.bin` , here is how to use these features. For the nearest neighbouring sentence feature, you need the model as well as a corpora in which you can search for the nearest neighbouring sentence to your input sentence. We use cosine distance as our distance metric. To do so, we use the command `nnSent` and the input should be 1 sentence per line:

```
./fasttext nnSent model.bin corpora [k] 
```
k is optional and is the number of nearest sentences that you want to output.     

For the analogiesSent, the user inputs 3 sentences A,B and C and finds a sentence from the corpora which is the closest to D in the A:B::C:D analogy pattern.
```
./fasttext analogiesSent model.bin corpora [k]
```

k is optional and is the number of nearest sentences that you want to output.     

# Unigram Embeddings 

For the purpose of generating word representations, we compared word embeddings obtained training sent2vec models with other word embedding models, including a novel method we refer to as CBOW char + word ngrams (`cbow-c+w-ngrams`). This method augments fasttext char augmented CBOW with word n-grams. You can see the full comparison of results in [*this paper*](https://www.aclweb.org/anthology/N19-1098). 

## Extracting Word Embeddings from Pre-Trained Models

If you have the Cython wrapper installed, some functionalities allow you to play with word embeddings obtained from `sent2vec` or `cbow-c+w-ngrams`:

```python
import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('model.bin') # The model can be sent2vec or cbow-c+w-ngrams
vocab = model.get_vocabulary() # Return a dictionary with words and their frequency in the corpus
uni_embs, vocab = model.get_unigram_embeddings() # Return the full unigram embedding matrix
uni_embs = model.embed_unigrams(['dog', 'cat']) # Return unigram embeddings given a list of unigrams
```  

Asking for a unigram embedding not present in the vocabulary will return a zero vector in case of sent2vec. The `cbow-c+w-ngrams` method will be able to use the sub-character ngrams to infer some representation. 

## Downloading Pre-Trained Models

Coming soon.

## Train a CBOW Character and Word Ngrams Model

Very similar to the sent2vec instructions. A plausible command would be:

    ./fasttext cbow-c+w-ngrams -input wiki_sentences.txt -output my_model -lr 0.05 -dim 300 -ws 10 -epoch 9 -maxVocabSize 750000 -thread 20 -numCheckPoints 20 -t 0.0001 -neg 5 -bucket 4000000 -bucketChar 2000000 -wordNgrams 3 -minn 3 -maxn 6

# References
When using this code or some of our pre-trained models for your application, please cite the following paper for sentence embeddings:

  Matteo Pagliardini, Prakhar Gupta, Martin Jaggi, [*Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features*](https://aclweb.org/anthology/N18-1049) NAACL 2018

```
@inproceedings{pgj2017unsup,
  title = {{Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features}},
  author = {Pagliardini, Matteo and Gupta, Prakhar and Jaggi, Martin},
  booktitle={NAACL 2018 - Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2018}
}
```

For word embeddings:

Prakhar Gupta, Matteo Pagliardini, Martin Jaggi, [*Better Word Embeddings by Disentangling Contextual n-Gram
Information*](https://www.aclweb.org/anthology/N19-1098) NAACL 2019

```
@inproceedings{DBLP:conf/naacl/GuptaPJ19,
  author    = {Prakhar Gupta and
               Matteo Pagliardini and
               Martin Jaggi},
  title     = {Better Word Embeddings by Disentangling Contextual n-Gram Information},
  booktitle = {{NAACL-HLT} {(1)}},
  pages     = {933--939},
  publisher = {Association for Computational Linguistics},
  year      = {2019}
}
```


