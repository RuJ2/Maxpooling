#include <iostream>
#include "../src/tensor.hpp"
#include "../src/maxPooling.hpp"
#include "../src/add.hpp"
using namespace std;

void show(Tensor& tensor)
{   
    cout << "dim: " << tensor.getDim() << endl;
    vector<int> shape = tensor.Shape();
    cout << "shape: [ ";
    for(auto i:shape)
        cout << i << " ";
    cout << "]" << endl;
}

int main(){
    vector<float> data{1,2,3,4,5,6,7,8};
    vector<int> shape{2,4};
    Tensor tensor(data, shape);
    cout << "Dim: " << tensor.getDim() << endl;
    cout << "Shape: ";
    for(auto i:tensor.Shape()) cout << i << " "; cout << endl;
    cout << "tensor[1,2]: " << tensor.at({1,2}) << endl;
    
    printf("-----------------\n");
    Tensor tensor1 = tensor;
    tensor1.newDim(3);
    cout << "New Shape: ";
    for(auto i:tensor1.Shape()) cout << i << " "; cout << endl;
    cout << "tensor[1,1,2]: " << tensor1.at({1,1,2}) << endl;

    printf("-----------------\n");
    Tensor tensor2 = add(tensor, tensor1);
    cout << "Dim: " << tensor2.getDim() << endl;
    cout << "Shape: ";
    for(auto i:tensor2.Shape()) cout << i << " "; cout << endl;
    cout << "tensor[1,1,2]: " << tensor2.at({1,1,2}) << endl;

    printf("-----------------\n");
    float* p3 = new float[2*2*5*5];
    for(int i=0; i<100; ++i) p3[i] = i;
    vector<int> shape_3 {2,2,5,5};
    Tensor tensor3(p3, shape_3);
    Tensor tensor3_out = maxPooling(tensor3);
    cout << "tensor3_out.shape: [ ";
    for(auto i:tensor3_out.Shape()) cout << i << " "; cout << "]\n";
    cout << tensor3_out.at({0,0,0,0}) << " " << tensor3_out.at({0,0,0,1}) << " " << tensor3_out.at({0,0,0,2}) << endl
         << tensor3_out.at({0,0,1,0}) << " " << tensor3_out.at({0,0,1,1}) << " " << tensor3_out.at({0,0,1,2}) << endl;

    return 0;
}