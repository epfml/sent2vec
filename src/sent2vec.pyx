import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector

#np.import_array()

cdef extern from "fasttext.h" namespace "fasttext":

    cdef cppclass FastText:
        FastText() except + 
        void loadModel(const string&)
        void textVector(string, vector[float]&)
        void textVectors(vector[string]&, int, vector[float]&)
        int getDimension()


cdef class Sent2vecModel:

    cdef FastText* _thisptr

    def __cinit__(self):
        self._thisptr = new FastText()

    def __dealloc__(self):
        del self._thisptr

    def __init__(self):
        pass  

    def get_emb_size(self):
        return self._thisptr.getDimension()
            
    def load_model(self, model_path):
        cdef string cmodel_path = model_path.encode('utf-8', 'ignore');
        self._thisptr.loadModel(cmodel_path)

    def embed_sentence(self, sentence):
        cdef string csentence = sentence.encode('utf-8', 'ignore');
        cdef vector[float] array;
        self._thisptr.textVector(csentence, array)
        return np.asarray(array)

    def embed_sentences(self, sentences, num_threads):
        if num_threads <= 0:
            num_threads = 1
        cdef vector[string] csentences;
        cdef int cnum_threads = num_threads;
        for s in sentences:
            csentences.push_back(s.encode('utf-8', 'ignore'));
        cdef vector[float] array;
        self._thisptr.textVectors(csentences, cnum_threads, array)
        return np.asarray(array).reshape(len(sentences), self.get_emb_size())

