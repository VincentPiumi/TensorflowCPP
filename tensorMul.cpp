// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    using RealType = float;
    
    const int depth = 10;
    const int height = 5;

    Tensor A(DataType::DT_FLOAT, TensorShape({depth, height}));
    auto Amap = A.tensor<RealType, 2>();
    Tensor B(DataType::DT_FLOAT, TensorShape({depth, height}));
    auto Bmap = B.tensor<RealType, 2>();
    
    for (int d = 0; d < depth; d++) {
	for (int h = 0; h < height; h++) {
	    Amap(d, h) = (d*1.0f + 4.)*0.5 + (h*1.0f + 5.)*0.5;
	    Bmap(d, h) = (d*1.0f + 5.)*0.5 + (h*1.0f + 6.)*0.5;
	}
    }
    
    Scope root = Scope::NewRootScope();
    auto mul = MatMul(root.WithOpName("mul"), A, B, MatMul::TransposeB(true));
    
    ClientSession session(root);
    std::vector<Tensor> out_map;
    TF_CHECK_OK(session.Run({mul}, &out_map));
    auto Omap = out_map[0].tensor<RealType, 2>();
    
    for (int d = 0; d < depth; d++) {
    	for (int h = 0; h < height; h++) {
	    std::cout << Omap(d, h) << ", ";
	}
	std::cout << std::endl;
    }
    
    return 0;
}
