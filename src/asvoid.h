template <class T> 
inline void *asvoid(std::vector<T> *buf) 
{ 
     std::vector<T>& tmp = *buf;
     return (void*)(&tmp[0]); 
} 