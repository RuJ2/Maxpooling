all: gtest_tensor.cpp gtest_add.cpp gtest_maxPooling.cpp
	g++ -std=c++11 gtest_tensor.cpp -o test_tensor -lgtest -lpthread
	g++ -std=c++11 gtest_add.cpp -o test_add -lgtest -lpthread
	g++ -std=c++11 gtest_maxPooling.cpp -o test_maxPooling -lgtest -lpthread -fopenmp

test_add: gtest_add.cpp
	g++ -std=c++11 gtest_add.cpp -o test_add -lgtest -lpthread
	
test_tensor: gtest_tensor.cpp
	g++ -std=c++11 gtest_tensor.cpp -o test_tensor -lgtest -lpthread

test_maxPooling: gtest_maxPooling.cpp
	g++ -std=c++11 gtest_maxPooling.cpp -o test_maxPooling -lgtest -lpthread -fopenmp

clean:
	rm test_tensor
	rm test_add
	rm test_maxPooling
