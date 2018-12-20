# sent2vec
TLDR: This library delivers numerical representations (features) for short texts or sentences, which can be used as input to any machine learning task later on. Think of it as an unsupervised version of [FastText](https://github.com/facebookresearch/fastText), and an extension of word2vec (CBOW) to sentences.

The method uses a simple but efficient unsupervised objective to train distributed representations of sentences. The algorithm outperforms the state-of-the-art unsupervised models on most benchmark tasks, and on many tasks even beats supervised models, highlighting the robustness of the produced sentence embeddings, see [*the paper*](https://arxiv.org/abs/1703.02507) for more details.

# Setup & Requirements
Our code builds upon [Facebook's FastText library](https://github.com/facebookresearch/fastText), see also their nice documentation and python interfaces.

To compile the library, simply run a `make` command.

# Generating Features from Pre-Trained Models

### Using the command-line interface

Given a pre-trained model `model.bin` (download links see below), here is how to generate the sentence features for an input text. To generate the features, use the `print-sentence-vectors` command and the input text file needs to be provided as one sentence per line:

```
./fasttext print-sentence-vectors model.bin < text.txt
```

This will output sentence vectors (the features for each input sentence) to the standard output, one vector per line.
This can also be used with pipes:

```
cat text.txt | ./fasttext print-sentence-vectors model.bin
```

### Directly from python

A Cython module allows you to keep the model in memory while inferring sentence embeddings:

```python
import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('model.bin')
emb = model.embed_sentence("once upon a time .") 
embs = model.embed_sentences(["first sentence .", "another sentence"])  
```
In order to compile and install the module globally, run the following from the `src` folder:

```
python setup.py build_ext
sudo pip install .
```
Text preprocessing (tokenization and lowercasing) is not handled by the module, check `wikiTokenize.py` for tokenization using NLTK and Stanford NLP. 

An alternative to the Cython module is using the python code provided in the `get_sentence_embeddings_from_pre-trained_models` notebook. It handles tokenization and can be given raw sentences, but does not keep the model in memory. 

#### Running inference with multiple processes

There is an 'inference' mode for loading the model in the Cython module, which loads the model's input matrix into a shared memory segment and doesn't load the output matrix, which is not needed for inference. This is an optimization for the usecase of running inference with multiple independent processes, which would otherwise each need to load a copy of the model into their address space. To use it:
```python
model.load_model('model.bin', inference_mode=True)
```

# Using Sentence level nearest neighbour search and analogies
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

### Downloading Pre-Trained Models

- [sent2vec_wiki_unigrams](https://drive.google.com/open?id=0B6VhzidiLvjSa19uYWlLUEkzX3c) 5GB (600dim, trained on english wikipedia)
- [sent2vec_wiki_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSaER5YkJUdWdPWU0) 16GB (700dim, trained on english wikipedia)
- [sent2vec_twitter_unigrams](https://drive.google.com/open?id=0B6VhzidiLvjSaVFLM0xJNk9DTzg) 13GB (700dim, trained on english tweets)
- [sent2vec_twitter_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSeHI4cmdQdXpTRHc) 23GB (700dim, trained on english tweets)
- [sent2vec_toronto books_unigrams](https://drive.google.com/open?id=0B6VhzidiLvjSOWdGM0tOX1lUNEk) 2GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))
- [sent2vec_toronto books_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSdENLSEhrdWprQ0k) 7GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))

(as used in the arXiv paper)

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

# Training New Models

To train a new sent2vec model, you first need some large training text file. This file should contain one sentence per line. The provided code does not perform tokenization and lowercasing, you have to preprocess your input data yourself, see above.

You can then train a new model. Here is one example of command:

    ./fasttext sent2vec -input wiki_sentences.txt -output my_model -minCount 8 -dim 700 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000

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
```

# References
When using this code or some of our pre-trained models for your application, please cite the following paper:

  Matteo Pagliardini, Prakhar Gupta, Martin Jaggi, [*Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features*](https://arxiv.org/abs/1703.02507) NAACL 2018

```
@inproceedings{pgj2017unsup,
  title = {{Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features}},
  author = {Pagliardini, Matteo and Gupta, Prakhar and Jaggi, Martin},
  booktitle={NAACL 2018 - Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2018}
}
```
