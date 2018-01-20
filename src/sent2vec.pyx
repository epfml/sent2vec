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
