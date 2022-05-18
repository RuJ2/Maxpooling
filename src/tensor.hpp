#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include<vector>
#include<numeric>
#include<memory>
using namespace std;

template<typename Dtype>
class Tensor{
public:
    Tensor() {}
    Tensor(vector<Dtype>& Data, vector<int>& shape);
    Tensor(Dtype*p, vector<int>& shape);
    ~Tensor(){}
    Tensor(const Tensor& tensor);
    Tensor<Dtype> operator=(Tensor& tensor);
    
    int getDim() { return ndim_; }
    vector<int> Shape() { return shape_; }
    Dtype* getPtr() {return pdata_.get();}  
    Dtype at(vector<int>&& p);  // Subscript access element
    Dtype at(vector<int>& p);
    void newDim(int newDim);

private:
    void init_shape(vector<int>& shape);
    void init_stride();

private:
    shared_ptr<Dtype> pdata_=nullptr;
    vector<int> shape_;
    vector<int> stride_;  // fast access through stride
    int ndim_ = 0;
};

template<typename Dtype>
Tensor<Dtype>::Tensor(vector<Dtype>& Data, vector<int>& shape){
    init_shape(shape);
    int max_size = accumulate(shape_.begin(), shape_.end(), 1 ,multiplies<int>());
    Dtype *p = new Dtype[max_size];
    
    // Input is truncated if exceed max_size
    int bound = Data.size() <= max_size ? Data.size() : max_size;
    // int bound = min(Data.size(), max_size);
    for(int i=0; i<bound; ++i)
        p[i] = Data[i];
    
    pdata_.reset(p);
};

template<typename Dtype>
Tensor<Dtype>::Tensor(Dtype*p, vector<int>& shape){
    init_shape(shape);
    pdata_.reset(p);
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const Tensor<Dtype>& tensor){
    ndim_ = tensor.ndim_;
    shape_ = tensor.shape_;
    init_stride();
    // Copy construction, The smart pointer points to data
    pdata_ = tensor.pdata_;
}

template<typename Dtype>
Dtype Tensor<Dtype>::at(vector<int>&& p){
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

// same to the right-value call
template<typename Dtype>
Dtype Tensor<Dtype>::at(vector<int>& p){
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

template<typename Dtype>
void Tensor<Dtype>::init_shape(vector<int>& shape){
    if(shape.empty()){
        perror("init_shape: empty\n");
        exit(-1);
    }
    for(int i=0; i<shape.size(); i++)
        if(shape[i]<=0){
            perror("init_shape: shape_size must be positive\n");
        }
    ndim_ = shape.size();
    shape_ = shape;
    init_stride();
}

template<typename Dtype>
void Tensor<Dtype>::init_stride(){
    stride_ = vector<int>(ndim_);
    int tmp = 1;
    for(int i=ndim_-1; i>=0; --i) {
        if(shape_[i]==1) stride_[i] = 0;  // Implement broadcast
        else    stride_[i] = tmp;
        tmp *= shape_[i];
    }
}   

template<typename Dtype>
void Tensor<Dtype>::newDim(int newDim){
    if(newDim<=ndim_){
        perror("newDim should be bigger\n");
        exit(-1);
    } 
    shape_.insert(shape_.begin(),newDim-ndim_, 1);
    ndim_ = newDim;
    init_stride();
}

#endif