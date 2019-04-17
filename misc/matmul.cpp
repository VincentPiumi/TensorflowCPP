// tensorflow/cc/example/matmul.cpp

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;
  
    Scope root = Scope::NewRootScope();

    auto A = Const(root, { {3.f, 2.f, 1.f}, {0.f, 1.f, 2.f}, {2.f, 0.f, 3.f} });
    auto B = Const(root, { {7.f, 6.f, 5.f}, {4.f, 5.f, 6.f}, {6.f, 4.f, 7.f} });

    auto mul = MatMul(root.WithOpName("mul"), A, B);

    std::vector<Tensor> out;
    ClientSession session(root);

    TF_CHECK_OK(session.Run({mul}, &out));
    LOG(INFO) << std::endl << out[0].matrix<float>();
}