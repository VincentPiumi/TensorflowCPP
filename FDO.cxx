#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/framework/ops.h>

#include "TFVector.hxx"
#include "Sum.hxx"
#include "Scale.hxx"

int main()
{
    int size = 50000;
    
    auto & root = TFConfig::scope();
    tensorflow::ClientSession session(root);

    TFVector<float> fijk1(size), fijk2(size), fijk3(size), fijk4(size);
    float c1 = 1.0/24.0;
    float c2 = -1.0/24.0;
    float c3 = 9.0/8.0;
    float c4 = -9.0/8.0;
    
    for (int i = 0; i < size; i++) {
	fijk1(i) = i * 1.0;
	fijk2(i) = i * 1.0;
	fijk3(i) = i * 1.0;
	fijk4(i) = i * 1.0;
    }  

    auto result = (c1 * fijk1 + c2 * fijk2 + c3 * fijk3 + c4 * fijk4)();

    std::vector<tensorflow::Tensor> output;
    TF_CHECK_OK(session.Run({result}, &output));
	
    return 0;
}
