// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;

    auto vec1 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f };
    auto vec2 = { 2.f, 1.f, 4.f, 5.f, 7.f, 1.f, 8.f, 6.f, 7.f, 1.f, 2.f };
    
    Scope root = Scope::NewRootScope();

    // auto A = Const(root, {vec1});
    // auto B = Const(root, {vec2});
    auto A = Variable(root, vec1, DT_FLOAT);
    auto B = Variable(root, vec2, DT_FLOAT);
    
    auto add = Add(root.WithOpName("add"), A, B);
 
    ClientSession session(root);
    std::vector<Tensor> out_add;
    Status s1 = session.Run({add}, &out_add);
    TF_CHECK_OK(session.Run({add}, &out_add));
    LOG(INFO) << std::endl << out_add[0].matrix<float>() << std::endl;
    
    return 0;
}


// 3.f, 3.f, 7.f, 9.f, 12.f, 7.f, 15.f, 14.f, 16.f, 11.f, 13.f
