#ifndef _MAXPOOLING_
#define _MAXPOOLING_
#include"tensor.hpp"
#include "omp.h"

const static int kernel_size = 3;
const static int pad = 1;
const static int stride = 2;

template<typename Dtype>
void maxPooling(Dtype*, Dtype*, int, int);

template<typename Dtype>
Tensor<Dtype> maxPooling(Tensor<Dtype> &tensor){
    if(tensor.getDim()<2){
        perror("maxPooling: Input dimension error");
        exit(-1);
    }

    vector<int> tensor_shape = tensor.Shape();
    int Image_row = tensor_shape[tensor.getDim()-2];
    int Image_col = tensor_shape.back();

    vector<int> out_shape = tensor_shape;
    int out_row = (Image_row-1)/stride + 1;
    int out_col = (Image_col-1)/stride + 1;
    out_shape[tensor.getDim()-2] = out_row;
    out_shape.back() = out_col;

    int max_size = accumulate(out_shape.begin(), out_shape.end(), 1 ,multiplies<int>());
    int Image_num = max_size/(out_row*out_col);
    Dtype *p = new Dtype[max_size];
    
#pragma omp parallel for
    for(int i=0; i<Image_num; i++)
    {
        // Gets the header address of input and output data
        maxPooling(tensor.getPtr()+i*Image_row*Image_col, p+i*out_row*out_col, Image_row, Image_col);
    }
    return Tensor<Dtype>(p, out_shape);
}

// overloading
template<typename Dtype>
void maxPooling(Dtype* srcImg, Dtype* dstImg, int row, int col){
    int out_row = (row-1)/stride + 1;
    int out_col = (col-1)/stride + 1;

#pragma omp parallel for
    for(int i=0; i<out_row; ++i)
    {
        int row_start = max(i*stride-1,0);
        int row_end = min(i*stride+1,row-1);

        for(int j=0; j<out_col; ++j)
        {
            int col_start = max(j*stride-1, 0);
            int col_end = min(j*stride+1, col-1);
            
            // if on the edge need pad 
            Dtype max_val = 0;
            if(row_start==0 || row_start==row || col_start==0 || col_start==col)
                max_val = pad;

            for(int _i=row_start; _i<=row_end; _i++){
                int loc = _i*col;
                for(int _j=col_start; _j<=col_end; _j++){
                    if(srcImg[loc+_j]>max_val)
                        max_val=srcImg[loc+_j];
                }
            }
            dstImg[i*out_col+j] = max_val;
        }
    }
}

#endif
