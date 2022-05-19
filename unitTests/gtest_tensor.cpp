#include <iostream>
#include "../src/tensor.hpp"
#include <gtest/gtest.h>
using namespace std;

namespace Test_tensor{

TEST(Tensor, DefaultConstructor){
    Tensor<int> tensor;
    EXPECT_EQ(tensor.getDim(), 0);
    EXPECT_TRUE(tensor.Shape().empty());
    EXPECT_TRUE(tensor.getPtr()==nullptr);
}

TEST(Tensor, ConstructWithDataAndShape){
    vector<int> data(100,0);
    for(int i=0; i<100; i++) data[i] = i;
    vector<int> shape{2,5,10};
    Tensor<int> tensor(data, shape);
    
    EXPECT_EQ(tensor.getDim(), 3);
    EXPECT_EQ(tensor.at({0,0,0}), 0);
    EXPECT_EQ(tensor.at({1,1,1}), 1*5*10+1*10+1);
    EXPECT_EQ(tensor.at({1,4,9}), 99);
    vector<int> tensor_shape = tensor.Shape();
    for(int i=0; i<tensor.getDim(); i++)
        EXPECT_EQ(tensor_shape[i], shape[i]);
}

TEST(Tensor, ConstructWithPointAndShape){
    float* p = new float[2*2*5*5];
    for(int i=0; i<100; ++i) p[i] = i;
    vector<int> shape {2,2,5,5};
    Tensor<float> tensor(p, shape);

    EXPECT_EQ(tensor.getDim(), 4);
    EXPECT_FLOAT_EQ(tensor.at({1,1,1,1}), 50+25+5+1);
}

TEST(Tensor, CopyConstruct){
    vector<int> data(100,0);
    for(int i=0; i<100; i++) data[i] = i;
    vector<int> shape{2,5,10};
    Tensor<int> tensor(data, shape);

    Tensor<int> tensor1(tensor);
    EXPECT_EQ(tensor.getDim(), tensor1.getDim());
    EXPECT_EQ(tensor.at({1,1,1}), tensor1.at({1,1,1}));
    vector<int> tensor_shape = tensor1.Shape();
    for(int i=0; i<tensor1.getDim(); i++)
        EXPECT_EQ(tensor_shape[i], shape[i]);
    EXPECT_EQ(tensor.getPtr(), tensor1.getPtr());
}

TEST(Tensor_func, at){
    vector<int> data(100,0);
    for(int i=0; i<100; i++) data[i] = i;
    vector<int> shape{2,5,10};
    Tensor<int> tensor(data, shape);

    EXPECT_NO_FATAL_FAILURE(tensor.at({1,1,1}));  // legal
    EXPECT_DEATH(tensor.at({1,1,100}), "");     // wrong idx
    EXPECT_DEATH(tensor.at({1,2,3,4}), "");     // beyond dim
    EXPECT_DEATH(tensor.at({-1,2,1}), "");  // negative shape
    EXPECT_DEATH(tensor.at({}), "");  // lack dim
    // EXPECT_EXIT(tensor.at({-1,2,1}), testing::ExitedWithCode(2), "");
}

TEST(Tensor_func, newDim){
    vector<int> data(100,0);
    vector<int> shape{2,5,10};
    Tensor<int> tensor(data, shape);

    EXPECT_DEATH(tensor.newDim(-1), "");  // negative dim
    EXPECT_DEATH(tensor.newDim(2), "");   // less dim
    EXPECT_NO_FATAL_FAILURE(tensor.newDim(4));  // legal
    EXPECT_EQ(tensor.getDim(), 4);   // 4==newDim
    EXPECT_EQ(tensor.Shape()[0], 1); // first dim == 1  shape: [1,2,5,10]

    // for broadcast
    EXPECT_EQ(tensor.at({0,1,1,1}), tensor.at({100,1,1,1}));
}

}

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
