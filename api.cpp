// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;
  
    Scope root = Scope::NewRootScope();

    auto A = Const(root, { {3.f, 2.f, 1.f}, {0.f, 1.f, 2.f}, {2.f, 0.f, 3.f} });
    auto B = Const(root, { {7.f, 6.f, 5.f}, {4.f, 5.f, 6.f}, {6.f, 4.f, 7.f} });
    auto T = Const(root, {{ {2.f}, {4.f}, {-6.f}, {1.f}, {-1.f}, {-9.f}, {0.f} }});
    
    auto add = Add(root.WithOpName("add"), A, B);
    auto sub = Subtract(root.WithOpName("sub"), A, B);
    auto abs = Abs(root.WithOpName("abs"), T);
    
    ClientSession session(root);
    
    std::vector<Tensor> out_sub;
    std::vector<Tensor> out_add;
    std::vector<Tensor> out_abs;
    
    Status s1 = session.Run({sub}, &out_sub);
    Status s2 = session.Run({add}, &out_add);
    Status s3 = session.Run({abs}, &out_abs);

    TF_CHECK_OK(session.Run({sub}, &out_sub));
    TF_CHECK_OK(session.Run({add}, &out_add));
    TF_CHECK_OK(session.Run({abs}, &out_abs));
    
    LOG(INFO) << std::endl << out_sub[0].matrix<float>() << std::endl;
    LOG(INFO) << std::endl << out_add[0].matrix<float>() << std::endl;
    LOG(INFO) << std::endl << out_abs[0].tensor<float, 3>() << std::endl;
    
    return 0;
}