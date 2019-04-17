// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
    using namespace tensorflow;
    using namespace tensorflow::ops;
  
    Scope root = Scope::NewRootScope();

    auto compxx = 1.f;
    auto compxy = 2.f;
    auto compxz = 3.f;
    auto compyx = 4.f;
    auto compyy = 5.f;
    auto compyz = 6.f;
    auto compzx = 7.f;
    auto compzy = 8.f;
    auto compzz = 9.f;

    auto Cauchy = Const(root,
    	{compxx, compxy, compxz, compyx, compyy, compyz, compzx, compzy, compzz},
    	{3,3});
    auto stress = Const(root, {1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f}, {3, 3});
    
    auto dot = Multiply(root.WithOpName("dot"), stress, Cauchy);
    
    ClientSession session(root);
    
    std::vector<Tensor> out;
    Status s1 = session.Run({dot}, &out);
 
    TF_CHECK_OK(session.Run({dot}, &out));
    LOG(INFO) << std::endl << out[0].matrix<float>() << std::endl;
    
    return 0;
}