// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    using RealType = float;
    
    const int depth = 5;
    const int height = 5;
    const int width = 4;

    Tensor A(DataType::DT_FLOAT, TensorShape({depth, height, width}));
    auto Amap = A.tensor<RealType, 3>();
    Tensor B(DataType::DT_FLOAT, TensorShape({depth, height, width}));
    auto Bmap = B.tensor<RealType, 3>();

    // Tensor out(DataType::DT_FLOAT, TensorShape({depth, height, width}));
    // auto out_map = out.tensor<float, 3>();
    
    for (int d = 0; d < depth; d++) {
	for (int h = 0; h < height; h++) {
	    for (int w = 0; w < width; w++) {
	        Amap(d, h, w) = 10;
	        Bmap(d, h, w) = 5;
	    }
	}
    }
    
    Scope root = Scope::NewRootScope();
    
    ClientSession session(root);
    std::vector<Tensor> out_map_, out_map;

    // auto add = Add(root.WithOpName("add"), A, B);    
    auto add2 = Multiply(root.WithOpName("add"), A, 3.f);
    
    TF_CHECK_OK(session.Run({add2}, &out_map));

    auto Omap = out_map[0].tensor<RealType, 3>();

    for (int d = 0; d < depth; d++) {
    	for (int h = 0; h < height; h++) {
    	    for (int w = 0; w < width; w++) {
    		std::cout << Omap(d, h, w) << ", ";
    	    }
    	    std::cout << " ";
    	}
    	std::cout << std::endl;
    }
    
    return 0;
}
