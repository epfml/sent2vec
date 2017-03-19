# sent2vec
This library provides unsupervised sentence representations (features) for short texts, which can be used as features as input to any machine learning task later on.

The method uses a simple but efficient unsupervised objective to train distributed representations of sentences. The algorithm outperforms the state-of-the-art unsupervised models on most benchmark tasks, and on many tasks even beats supervised models, highlighting the robustness of the produced sentence embeddings, see [*the paper*](https://arxiv.org/abs/1703.02507) for more details.

TODO: add more details?

TODO: add code


# Setup & Requirements
Our code builds upon [Facebook's FastText library](https://github.com/facebookresearch/fastText), see also their nice documentation and python interfaces.


# Generating Features from Pre-Trained Models
Given some existing model `model.bin` and arbitrary input text (one sentence per line), here is how to generate the sentence features:

```
$ print-vectors model.bin < text.txt
```

This will output word vectors to the standard output, one vector per line.
This can also be used with pipes:

```
$ cat text.txt | print-vectors model.bin
```

# Training New Models
Training method:

```
$ sent2vec -input train.txt -output model

The following arguments are mandatory:
  -input              training file path
  -output             output file path

The following arguments are optional:
  -lr                 learning rate [0.2]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                size of word vectors [100]
  -ws                 size of the context window [5]
  -epoch              number of epochs [5]
  -minCount           minimal number of word occurences [5]
  -minCountLabel      minimal number of label occurences [0]
  -neg                number of negatives sampled [10]
  -wordNgrams         max length of word ngram [2]
  -loss               loss function {ns, hs, softmax} [ns]
  -bucket             number of buckets [2000000]
  -minn               min length of char ngram [0]
  -maxn               max length of char ngram [0]
  -thread             number of threads [2]
  -t                  sampling threshold [0.0001]
  -label              labels prefix [__label__]
  -dropoutK           number of ngrams dropped when training a sent2vec model [2]
  -verbose            verbosity level [2]
  -pretrainedVectors  pretrained word vectors for supervised learning []
  -saveOutput         whether output params should be saved [0]
```

An example of command:

    ./fasttext sent2vec -input wiki_sentences.txt -output my_model -minCount 8 -dim 700 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000
    
The input file consists in a simple textfile with one sentence per line. 

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

