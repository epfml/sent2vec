# sent2vec
TLDR: This library delivers numerical representations (features) for short texts or sentences, which can be used as input to any machine learning task later on. Think of it as an unsupervised version of [FastText](https://github.com/facebookresearch/fastText), and an extension of CBOW to sentences.

The method uses a simple but efficient unsupervised objective to train distributed representations of sentences. The algorithm outperforms the state-of-the-art unsupervised models on most benchmark tasks, and on many tasks even beats supervised models, highlighting the robustness of the produced sentence embeddings, see [*the paper*](https://arxiv.org/abs/1703.02507) for more details.

# Setup & Requirements
Our code builds upon [Facebook's FastText library](https://github.com/facebookresearch/fastText), see also their nice documentation and python interfaces.

To compile the library, simply run a `make` command from the `src` folder.

# Generating Features from Pre-Trained Models
Given a pre-trained model `model.bin` (download links see below), here is how to generate the sentence features for an input text. The input text file needs to be provided as one sentence per line.

```
./fasttext print-vectors model.bin < text.txt
```

This will output sentence vectors (the features for each input sentence) to the standard output, one vector per line.
This can also be used with pipes:

```
cat text.txt | ./fasttext print-vectors model.bin
```

Before using, make sure to apply the same text preprocessing pipeline as used for training to your input text! TODO: link for preprocessing as in pre-trained models.

### Downloading Pre-Trained Models

- [sent2vec_wiki_unigrams](https://drive.google.com/uc?export=download&confirm=FHHw&id=0BwblUWuN_Bn9akZpdVg0Qk8zbGs) 5GB (600dim, trained on wikipedia)
- [sent2vec_wiki_bigrams](https://drive.google.com/uc?export=download&confirm=IcCE&id=0BwblUWuN_Bn9RURIYXNKeE5qS1U) 16GB (700dim, trained on wikipedia)
- [sent2vec_twitter_unigrams](https://drive.google.com/uc?export=download&confirm=D2U1&id=0BwblUWuN_Bn9RkdEZkJwQWs4WmM) 13GB (700dim, trained on tweets)
- [sent2vec_twitter_bigrams](https://drive.google.com/uc?export=download&confirm=BheQ&id=0BwblUWuN_Bn9VTEyUzA2ZFNmVWM) 23GB (700dim, trained on tweets)

(as used in the arXiv paper)

# Training New Models

To train a new sent2vec model, you first need some large training text file. This file should contain one sentence per line. The provided code does not perform tokenization and lowercasing, you have to preprocess your input data yourself.

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

# Tokenizing
In order to get the tokenized sentences for tweets, run 

```
python3 tweetTokenize.py <tweets_folder> <dest_folder> <num_process>
```

to get the sentences in a tokenized form.

For tokenized sentences for wiki, run
```
python3 wikiTokenize.py corpora > destinationFile
```
NOTE : Set the BASE_SNLP_PATH and SNLP_TAGGER_JAR correctly for wikiTokenize.py .

SNLP_TAGGER_JAR is the address of stanford-postagger.jar which you can download from http://www.java2s.com/Code/Jar/s/Downloadstanfordpostaggerjar.htm

# References
When using this code or some of our pre-trained models for your application, please cite the following paper:

  Matteo Pagliardini, Prakhar Gupta, Martin Jaggi, [*Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features*](https://arxiv.org/abs/1703.02507) arXiv

```
@article{pgj2017unsup,
  title = {{Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features}},
  author = {Pagliardini, Matteo and Gupta, Prakhar and Jaggi, Martin},
  journal = {arXiv},
  eprint = {1703.02507},
  eprinttype = {arxiv},
  eprintclass = {cs.CL},
  year = {2017}
}
```
