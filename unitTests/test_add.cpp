#include <iostream>
#include "../src/tensor.hpp"
#include "../src/add.hpp"
#include <gtest/gtest.h>
using namespace std;

namespace Test_add{
    
TEST(add, SimpleAdd){
    int *pdata1 = new int[6];
    for(int i=0; i<6; ++i) pdata1[i] = i;
    vector<int> shape{1,2,3};
    Tensor<int> tensor1(pdata1, shape);

    int *pdata2 = new int[6];
    for(int i=0; i<6; ++i) pdata2[i] = i+1;
    Tensor<int> tensor2(pdata2, shape);

    Tensor<int> tensor3 = add(tensor1, tensor2);
    // same shape
    EXPECT_EQ(tensor3.getDim(), 3);
    for(int i=0; i<3; i++)
        EXPECT_EQ(tensor3.Shape()[i], shape[i]);
    // value verify
    EXPECT_EQ(tensor3.at({0,0,0}), 0+1);
    EXPECT_EQ(tensor3.at({0,1,2}), 5+6);
}

TEST(add, BroadcastAdd){
    vector<float> data = {1,2,3,4,5,6};
    vector<int> shape1{  6,1};
    vector<int> shape2{3,1,2};
    Tensor<float> tensor1(data, shape1);
    Tensor<float> tensor2(data, shape2);
    Tensor<float> tensor3 = add(tensor1, tensor2);
        
    // tensor1 and tensor2 don't change
    EXPECT_EQ(tensor1.getDim(), 2);
    EXPECT_EQ(tensor1.Shape()[0], 6);
    EXPECT_FLOAT_EQ(tensor1.at({3,0}), 4);

    // tensor3
    // shape validation
    EXPECT_EQ(tensor3.getDim(), 3);
    vector<int> shape_expected {3,6,2};
    for(int i=0; i<3; ++i)
        EXPECT_EQ(tensor3.Shape()[i], shape_expected[i]);
    // data validation
    EXPECT_EQ(tensor3.at({1,4,1}), tensor1.at({4,0})+tensor2.at({1,0,1}));
}

TEST(add, FloatAdd){
    float *pdata1 = new float[6];
    for(int i=0; i<6; ++i) pdata1[i] = i;
    vector<int> shape{1,2,3};
    Tensor<float> tensor1(pdata1, shape);
    Tensor<float> tensor2(pdata1, shape);
    Tensor<float> tensor3 = add(tensor1, tensor2);
    EXPECT_FLOAT_EQ(tensor3.at({0,1,2}), 5.0+5.0);
}

TEST(add, IfCanBroadcastAdd){
    vector<float> data{1,2,3,4,5,6};
    vector<int> shape1{  6,1};
    vector<int> shape2{  3,2};
    vector<int> shape3{1,6,5};
    Tensor<float> tensor1(data, shape1);
    Tensor<float> tensor2(data, shape2);
    Tensor<float> tensor3(data, shape3);  // will padding 0

    EXPECT_DEATH(add(tensor1, tensor2), "");   // No 1 and 6!=3 in shape[0]
    EXPECT_NO_FATAL_FAILURE(add(tensor1, tensor3));  // legal
}

TEST(add, last){

}

}

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}