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

    std::vector<Tensor> mm, sm;
    Tensor t1(DataType::DT_FLOAT, TensorShape({depth, height}));
    Tensor t2(DataType::DT_FLOAT, TensorShape({depth, height}));
    Tensor t3(DataType::DT_FLOAT, TensorShape({depth, height}));

    auto t1map = t1.tensor<RealType, 2>();
    auto t2map = t2.tensor<RealType, 2>();
    auto t3map = t3.tensor<RealType, 2>();

    for (int d = 0; d < depth; d++) {
	for (int h = 0; h < height; h++) {
	    t1map(d, h) = d + (h + 1.f);
	    t2map(d, h) = t1map(d, h) * 2.f;
        t3map(d, h) = t2map(d, h) * 2.f;
    }
    }

    Output o = Add(root.WithOpName("matmul"), t1, t2);
    Input in = Input(o);
    Input add = Add(root.WithOpName("matmul"),
                      in,
                      t3);

    Output matmul(add.node());

    TF_CHECK_OK(session.Run({matmul}, &mm));
    auto mmmap = mm[0].tensor<RealType, 2>();    

    for (int d = 0; d < depth; d++) {
    	for (int h = 0; h < height; h++) {
    	    std::cout.width(4);
    	    std::cout << mmmap(d, h);
    	}
    	std::cout.width(4);
    	std::cout << std::endl;
    }

    return 0;
}
