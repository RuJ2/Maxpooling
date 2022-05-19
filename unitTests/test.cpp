#include <iostream>
#include "../src/tensor.hpp"
#include "../src/maxPooling.hpp"
#include "../src/add.hpp"
#include <omp.h>
using namespace std;

template<typename Dtype>
void show(Tensor<Dtype>& tensor)
{   
    cout << "dim: " << tensor.getDim() << endl;
    vector<int> shape = tensor.Shape();
    cout << "shape: [ ";
    for(auto i:shape)
        cout << i << " ";
    cout << "]" << endl;
}

int main(){
    vector<float> data = {1,2,3,4,5,6};
    vector<int> shape1{  6,1};
    vector<int> shape2{3,1,2};
    Tensor<float> tensor1(data, shape1);
    Tensor<float> tensor2(data, shape2);
    
    Tensor<float> tensor3 = add(tensor1, tensor2);
    
    for(int i=0; i<3; ++i)
        cout << tensor3.Shape()[i] << " ";
    cout << endl;
    show(tensor3);
    cout << tensor3.at({0,0,0}) << " " << tensor3.at({0,0,1}) << endl;

    cout << "CPU_NUM: " << omp_get_num_procs() << endl;
#pragma omp parallel for
    for(int i=0; i<10; ++i)
        cout << i << " ";
    cout << endl;
    // vector<float> data{1,2,3,4,5,6,7,8};
    // vector<int> shape{2,4};
    // Tensor<float> tensor(data, shape);
    // show(tensor);
    // cout << "tensor[1,2]: " << tensor.at({1,2}) << endl;
    
    // printf("-----------------\n");
    // Tensor<float> tensor1 = tensor;
    // tensor1.newDim(3);
    // show(tensor1);
    // cout << "tensor[1,1,2]: " << tensor1.at({1,1,2}) << endl;

    // printf("-----------------\n");
    // Tensor<float> tensor2 = add(tensor, tensor1);
    // show(tensor2);
    // cout << "tensor[1,1,2]: " << tensor2.at({1,1,2}) << endl;

    // printf("-----------------\n");
    // float* p3 = new float[2*2*5*5];
    // for(int i=0; i<100; ++i) p3[i] = i;
    // vector<int> shape_3 {2,2,5,5};
    // Tensor<float> tensor3(p3, shape_3);
    // Tensor<float> tensor3_out = maxPooling(tensor3);
    // show(tensor3);
    // printf("----------\n");
    // show(tensor3_out);
    // cout << tensor3_out.at({0,0,0,0}) << " " << tensor3_out.at({0,0,0,1}) << " " << tensor3_out.at({0,0,0,2}) << endl
    //      << tensor3_out.at({0,0,1,0}) << " " << tensor3_out.at({0,0,1,1}) << " " << tensor3_out.at({0,0,1,2}) << endl;

    return 0;
}
