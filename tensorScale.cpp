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
    const RealType scalar = 10.f;

    Scope root = Scope::NewRootScope();
    ClientSession session(root);
    std::vector<Tensor> out_map;
    
    Tensor M(DataType::DT_FLOAT, TensorShape({depth, height}));
    auto Mmap = M.tensor<RealType, 2>();
    Tensor vec(DataType::DT_FLOAT, TensorShape({depth}));
    auto vecmap = vec.tensor<RealType, 1>();
    
    for (int d = 0; d < depth; d++) {
	for (int h = 0; h < height; h++) {
	    Mmap(d, h) = (d*1.0f + 4.)*0.5 + (h*1.0f + 5.)*0.5;
	}
	vecmap(d) = (d*1.0f + 4.)*0.5;
    }


    // ---- MATRIX CASE
    // auto mul = Multiply(root.WithOpName("mul"), M, scalar);
    // TF_CHECK_OK(session.Run({mul}, &out_map));
    // auto Omap = out_map[0].tensor<RealType, 2>();
	
    // for (int d = 0; d < depth; d++) {
    // 	for (int h = 0; h < height; h++) {
    // 	    std::cout.width(4);
    // 	    std::cout << Omap(d, h);
    // 	}
    // 	std::cout.width(4);
    // 	std::cout << std::endl;
    // }

    // ---- VECTOR CASE
    auto mul = Multiply(root.WithOpName("mul"), vec, scalar);
    TF_CHECK_OK(session.Run({mul}, &out_map));
    auto Omap = out_map[0].tensor<RealType, 1>();

    for (int d = 0; d < depth; d++) {
	std::cout.width(4);
	std::cout << Omap(d);
    }
	
    
    return 0;
}