#ifndef _ADD_HPP_
#define _ADD_HPP_

#include "tensor.hpp"

Tensor add(Tensor a, Tensor b){
    if(a.getDim()==0 || b.getDim()==0 ){
        perror("add: Input tensor dimension can't be empty\n");
        exit(-1);
    }
    int NewDim = max(a.getDim(), b.getDim());
    if(a.getDim()<NewDim) a.newDim(NewDim);
    if(b.getDim()<NewDim) b.newDim(NewDim);
    vector<int> Shape_a = a.Shape();
    vector<int> Shape_b = b.Shape();
    vector<int> Shape_res(NewDim);
    for(int i=NewDim-1; i>=0; --i){
        if(Shape_a[i]==1 || Shape_a[i]==1 || Shape_a[i]==Shape_b[i]){
            Shape_res[i] = max(Shape_a[i], Shape_b[i]);
        }
        else{
            perror("Dissatisfy the add condition\n");
            exit(0);
        }
    }
    int max_size = accumulate(Shape_res.begin(), Shape_res.end(), 1 ,multiplies<int>());
    float *p = new float[max_size];
    
    // Subscript index increment assignment
    vector<int> place(NewDim, 0);
    for(int i=0; i<max_size; ++i)
    {
        p[i] = a.at(place) + b.at(place);
        for(int i=NewDim-1; i>=0; i--)
        {
            if(++place[i]==Shape_res[i]){
                place[i]=0;
                continue;
            }
            else break;
        }
    }
    return Tensor(p,Shape_res);
}

#endif