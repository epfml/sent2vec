
#include "fasttext.h"

using namespace fasttext;

extern "C" void* loadModel(char* path)
{
    FastText* fastText = new FastText();
    (*fastText).loadModel(std::string(path));
    return fastText;
}

extern "C" void freeModel(void* modelPtr)
{
    FastText* model = (FastText*)modelPtr;
    free(model);
}

extern "C" int getDimension(void* modelPtr)
{
    FastText* model = (FastText*)modelPtr;
    return (*model).getDimension();
}

extern "C" void getSentenceVector(
    void* modelPtr,
    char* input,
    float* svec
)
{
    FastText* model = (FastText*)modelPtr;
    int dim = (*model).getDimension();
    std::vector<std::string> strList(1);
    strList[0] = std::string(input);
    std::vector<float> vector(dim);
    (*model).textVectors(strList, 1, vector);
    memcpy(svec, &vector[0], vector.size() * sizeof(float));
}
