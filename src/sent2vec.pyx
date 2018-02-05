import numpy as np
cimport numpy as cnp 

from libcpp.string cimport string
from libcpp.vector cimport vector

#from libc.stdlib cimport free
#from cpython cimport PyObject, Py_INCREF

cnp.import_array()

cdef extern from "fasttext.h" namespace "fasttext":

    cdef cppclass FastText:
        FastText() except + 
        void loadModel(const string&)
        void textVector(string, vector[float]&)
        void textVectors(vector[string]&, int, vector[float])#&)
        int getDimension()


cdef extern from "asvoid.h": 
     void *asvoid(vector[float] *buf)


class stdvector_base: 
    pass 


cdef class vector_wrapper: 
    cdef: 
        vector[float] *buf 

    def __cinit__(vector_wrapper self, n): 
        self.buf = NULL 

    def __init__(vector_wrapper self, cnp.intp_t n): 
        self.buf = new vector[float](n) 

    def __dealloc__(vector_wrapper self): 
        if self.buf != NULL: 
            del self.buf 

    def asarray(vector_wrapper self, cnp.intp_t n): 
        """ 
        Interpret the vector as np.ndarray without 
        copying the data. 
        """ 
        base = stdvector_base() 
        intbuf = <cnp.uintp_t> asvoid(self.buf) 
        dtype = np.dtype(np.float32) 
        base.__array_interface__ = dict( 
            data = (intbuf, False), 
            descr = dtype.descr, 
            shape = (n,), 
            strides = (dtype.itemsize,), 
            typestr = dtype.str, 
            version = 3, 
        ) 
        base.vector_wrapper = self 
        return np.asarray(base) 
        

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

    def embed_sentences(self, sentences, num_threads=1):
        if num_threads <= 0:
            num_threads = 1
        cdef vector[string] csentences
        cdef int cnum_threads = num_threads
        for s in sentences:
            csentences.push_back(s.encode('utf-8', 'ignore'));
        cdef vector_wrapper array 
        w = vector_wrapper(len(sentences) * self.get_emb_size())
        self._thisptr.textVectors(csentences, cnum_threads, w.buf[0])
        final = w.asarray(len(sentences) * self.get_emb_size()) 
        return final.reshape(len(sentences), self.get_emb_size())
