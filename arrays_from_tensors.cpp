/// \brief Trying to create Eigen tensors, and Tensorflow tensors from
/// simple C++ arrays in different alignment (row-major, col-major, 3d etc.).
/// \author David Stutz
/// \file tensors_from_array.cc
     
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
     
using namespace tensorflow;
     
/// \brief Trying to create Eigen/Tensorflow tensors from arrays.
/// \param argc
/// \param argv
/// \return 
int main(int argc, char** argv) {
      
    const int batch_size = 1;
    const int depth = 5;
    const int height = 5;
    const int width = 5;
    const int channels = 3;
    
    // trying to create and fill a Tensorflow tensor, with at least 4 dimensions
    // to see how the data is stored in memory;
    // hopefully able to convert it to the Eigen tensor and from there to an array.
    Tensor tensor(DataType::DT_INT32, TensorShape({batch_size, depth, height, width, channels}));
      
    // get underlying Eigen tensor
    auto tensor_map = tensor.tensor<int, 5>();
      
    // fill and print the tensor
    for (int n = 0; n < batch_size; n++) {
        for (int d = 0; d < depth; d++) {
	    std::cout << d << " --" << std::endl;
	    for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
		    for (int c = 0; c < channels; c++) {
			tensor_map(n, d, h, w, c) = (((n*depth + d)*height + h)*width + w)*channels + c;
			std::cout << tensor_map(n, d, h, w, c) << ",";
		    }
		    std::cout << " ";
		}
		std::cout << std::endl;
	    }
        }
    }
      
    // get the underlying array
    auto array = tensor_map.data();
    int* int_array = static_cast<int*>(array);
      
    // try to print the same to see the data layout
    for (int n = 0; n < batch_size; n++) {
        for (int d = 0; d < depth; d++) {
	    std::cout << d << " --" << std::endl;
	    for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
		    for (int c = 0; c < channels; c++) {
			std::cout << int_array[(((n*depth + d)*height + h)*width + w)*channels + c] << ",";
		    }
		    std::cout << " ";
		}
		std::cout << std::endl;
	    }
        }
    }
      
    return 0;
}