#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include<vector>
#include<numeric>
#include<memory>
using namespace std;

class Tensor{
public:
    Tensor(vector<float>& Data, vector<int>& shape);
    Tensor(float*p, vector<int>& shape);
    ~Tensor(){}
    Tensor(const Tensor& tensor);
    
    void init_shape(vector<int>& shape);
    void init_stride();
    int getDim() { return ndim_; }
    float* getPtr() {return pdata_.get();}
    vector<int> Shape() { return shape_; }
    float at(vector<int>&& p);
    float at(vector<int>& p);
    void newDim(int newDim);

private:
    shared_ptr<float> pdata_=nullptr;
    vector<int> shape_;
    vector<int> stride_;  // fast access through stride
    int ndim_;
};

Tensor::Tensor(vector<float>& Data, vector<int>& shape){
    init_shape(shape);
    int max_size = accumulate(shape_.begin(), shape_.end(), 1 ,multiplies<int>());
    float *p = new float[max_size];
    
    // Input is truncated if exceed max_size
    int bound = Data.size() <= max_size ? Data.size() : max_size;
    // int bound = min(Data.size(), max_size);
    for(int i=0; i<bound; ++i)
        p[i] = Data[i];
    
    pdata_.reset(p);
}

Tensor::Tensor(float*p, vector<int>& shape){
    init_shape(shape);
    pdata_.reset(p);
}

Tensor::Tensor(const Tensor& tensor){
    ndim_ = tensor.ndim_;
    shape_ = tensor.shape_;
    init_stride();
    // Copy construction, The smart pointer points to data
    pdata_ = tensor.pdata_;
}

float Tensor::at(vector<int>&& p){
    if(p.size()!=shape_.size())
    {
        perror("at: Input dimensions do not match\n");
        exit(-1);
    }
    else{
        int idx = 0;
        for(int i=0; i<ndim_; ++i){
            if(p[i]<0 || p[i]>=shape_[i]&&shape_[i]!=1) {
                perror("at: Incorrect subscript input\n");
                exit(-1);
            }
            else{
                idx += p[i]*stride_[i];
            }
        }
        return pdata_.get()[idx];
    }
}

float Tensor::at(vector<int>& p){
    if(p.size()!=shape_.size())
    {
        perror("at: Input dimensions do not match\n");
        exit(-1);
    }
    else{
        int idx = 0;
        for(int i=0; i<ndim_; ++i){
            if(p[i]<0 || p[i]>=shape_[i]&&shape_[i]!=1) {
                perror("at: Incorrect subscript input\n");
                exit(-1);
            }
            else{
                idx += p[i]*stride_[i];
            }
        }
        return pdata_.get()[idx];
    }
}

void Tensor::init_shape(vector<int>& shape){
    ndim_ = shape.size();
    shape_ = shape;
    init_stride();
}

void Tensor::init_stride(){
    stride_ = vector<int>(ndim_);
    int tmp = 1;
    for(int i=ndim_-1; i>=0; --i) {
        if(shape_[i]==1) stride_[i] = 0;
        else    stride_[i] = tmp;
        tmp *= shape_[i];
    }
}   

void Tensor::newDim(int newDim){
    if(newDim<=ndim_) return;
    shape_.insert(shape_.begin(),newDim-ndim_, 1);
    ndim_ = newDim;
    init_stride();
}

#endif