# Maxpooling

The repository is for the tensor's maxPooling function, for two tensor's addition function satisfy the broadcast mechanism.

### Directory Structure
There are two directory include source files and unittests files.
- In src directory contains three .hpp files
  - Tensor class
  - add function
  - maxPooling function
- In unittests directory contains three gtest files and a simple test.cpp
  - gtest_tensor.cpp
  - gtest_add.cpp
  - gtest_maxPooling.cpp

### Getting Started
- Make sure your compiler compliant with the **C++11** standard or newer.
- Make sure you have installed and configured **googletest**. [reference](https://www.cnblogs.com/galaxy-hao/p/13171340.html)

1. ` cd unitTests && make `
2. ` ./test_tensor; ./test_add; ./test_maxPooling `
