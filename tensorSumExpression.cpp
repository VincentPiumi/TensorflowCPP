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
    
    Scope root = Scope::NewRootScope();
    ClientSession session(root);

    std::vector<Tensor> sum1_res, sum2_res;
    Tensor t1(DataType::DT_FLOAT, TensorShape({depth, height}));
    Tensor t2(DataType::DT_FLOAT, TensorShape({depth, height}));
    Tensor t3(DataType::DT_FLOAT, TensorShape({depth, height}));
    
    auto t1map = t1.tensor<RealType, 2>();
    auto t2map = t2.tensor<RealType, 2>();
    auto t3map = t3.tensor<RealType, 2>();
    
    auto holder1 = Placeholder(root, DataType::DT_FLOAT);
    auto holder2 = Placeholder(root, DataType::DT_FLOAT);
    auto holder3 = Placeholder(root, DataType::DT_FLOAT);
    
    for (int d = 0; d < depth; d++) {
	for (int h = 0; h < height; h++) {
	    t1map(d, h) = d + (h + 1.f);
	    t2map(d, h) = t1map(d, h) * 2.f;
	    t3map(d, h) = t2map(d, h) * 4.f;
	}
    }    
    
    auto sum1 = Add(root.WithOpName("sum1"), holder1, holder2);
    auto sum2 = Add(root.WithOpName("sum2"), holder3, t3);

    TF_CHECK_OK(session.Run({{holder1, t1}, {holder2, t2}}, {sum1}, &sum1_res));
    TF_CHECK_OK(session.Run({{holder3, sum1_res[0]}}, {sum2}, &sum2_res));

    auto totalmap = sum2_res[0].tensor<RealType, 2>();

    for (int d = 0; d < depth; d++) {
    	for (int h = 0; h < height; h++) {
    	    std::cout.width(4);
    	    std::cout << totalmap(d, h);
    	}
    	std::cout.width(4);
    	std::cout << std::endl;
    }

    return 0;
}
