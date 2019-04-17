// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;
    using RealType = float;
    
    const int depth = 3;
    const int height = 2;
    const float scal = 10.f;

    Scope root = Scope::NewRootScope();
    Scope matrix = root.NewSubScope("matrix");
    Scope scalar = root.NewSubScope("scalar");
    ClientSession session(root);

    std::vector<Tensor> mm, sm;    
    Tensor t1(DataType::DT_FLOAT, TensorShape({depth, height}));
    Tensor t2(DataType::DT_FLOAT, TensorShape({depth, height}));
    
    auto t1map = t1.tensor<RealType, 2>();
    auto t2map = t2.tensor<RealType, 2>();
    auto t3 = Placeholder(root, DataType::DT_FLOAT);
    
    for (int d = 0; d < depth; d++) {
	for (int h = 0; h < height; h++) {
	    t1map(d, h) = d + (h + 1.f);
	    t2map(d, h) = t1map(d, h) * 2.f;
	}
    }
    
    auto matmul = MatMul(matrix.WithOpName("matmul"), t1, t2, MatMul::TransposeB(true));
    auto scamul = Multiply(scalar.WithOpName("scamul"), t3, scal);

    TF_CHECK_OK(session.Run({matmul}, &mm));
    TF_CHECK_OK(session.Run({{t3, mm[0]}}, {scamul}, &sm));
    auto mmmap = mm[0].tensor<RealType, 2>();    
    auto smmap = sm[0].tensor<RealType, 2>();

    for (int d = 0; d < depth; d++) {
    	for (int h = 0; h < height; h++) {
    	    std::cout.width(4);
    	    std::cout << smmap(d, h);
    	}
    	std::cout.width(4);
    	std::cout << std::endl;
    }

    return 0;
}
