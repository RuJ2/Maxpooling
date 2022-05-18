#include <iostream>
#include "../src/tensor.hpp"
#include "../src/maxPooling.hpp"
#include <gtest/gtest.h>
using namespace std;

namespace Test_maxPooling{

TEST(maxPooling, simpleTest){
    float* p = new float[2*2*5*5];
    for(int i=0; i<100; ++i) p[i] = i;
    vector<int> shape_ {2,2,5,5};
    Tensor<float> tensor(p, shape_);
    Tensor<float> tensor_out = maxPooling(tensor);
    // shape validation
    EXPECT_EQ(tensor_out.getDim(), 4);
    vector<int> shape_out_expected {2,2,3,3};
    for(int i=0; i<4; i++)
        EXPECT_EQ(tensor_out.Shape()[i], shape_out_expected[i]);
    // value validation
    /*
    0, 1, 2, 3, 4           6, 8, 9
    5, 6, 7, 8, 9     ->    16,18,19
    10,11,12,13,14          ...
    15,16,17,18,19
    ...
    */
    EXPECT_EQ(tensor_out.at({0,0,0,0}), 6);
    EXPECT_EQ(tensor_out.at({0,0,0,1}), 8);
    EXPECT_EQ(tensor_out.at({0,0,0,2}), 9);
    EXPECT_EQ(tensor_out.at({0,0,1,0}), 16);
    EXPECT_EQ(tensor_out.at({1,1,2,2}), 99);
}

TEST(maxPooling, LegalInput){
    float* p = new float[100];
    vector<int> shape_ {100};
    Tensor<float> tensor(p, shape_);
    EXPECT_DEATH(maxPooling(tensor),""); // shape less two

    float* p1 = new float[1];
    vector<int> shape1 {1,1};  
    Tensor<float> tensor1(p1, shape1);
    // just single value, But that's right, because shape_dim >= 2
    EXPECT_NO_FATAL_FAILURE(maxPooling(tensor1));
    Tensor<float> tensor_out = maxPooling(tensor1);
    EXPECT_EQ(tensor_out.at({0,0}), 1);  // padding is 1;
}

}

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}