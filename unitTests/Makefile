all: gtest_tensor.cpp gtest_add.cpp gtest_maxPooling.cpp
	g++ gtest_tensor.cpp -o test_tensor -lgtest -lpthread
	g++ gtest_add.cpp -o test_add -lgtest -lpthread
	g++ gtest_maxPooling.cpp -o test_maxPooling -lgtest -lpthread

test_add: gtest_add.cpp
	g++ gtest_add.cpp -o test_add -lgtest -lpthread
	
test_tensor: gtest_tensor.cpp
	g++ gtest_tensor.cpp -o test_tensor -lgtest -lpthread

test_maxPooling: gtest_maxPooling.cpp
	g++ gtest_maxPooling.cpp -o test_maxPooling -lgtest -lpthread

clean:
	rm test_tensor
	rm test_add
	rm test_maxPooling
