test_add: test_add.cpp
	g++ test_add.cpp -o test_add -lgtest -lpthread
	
all: test_tensor.cc test_add.cpp test_maxPooling.cpp
	g++ test_tensor.cc -o test_tensor -lgtest -lpthread
	g++ test_add.cpp -o test_add -lgtest -lpthread
	g++ test_maxPooling.cpp -o test_maxPooling -lgtest -lpthread

test_tensor: test_tensor.cc
	g++ test_tensor.cc -o test_tensor -lgtest -lpthread


test_maxPooling: test_maxPooling.cpp
	g++ test_maxPooling.cpp -o test_maxPooling -lgtest -lpthread

clean:
	rm test_tensor
	rm test_add
	rm test_maxPooling