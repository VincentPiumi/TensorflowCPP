/// Some array conversion tools.
/// \author David Stutz
template<typename T, int NDIMS>
class TensorConversion {
public:
      
    /// Access the underlying data pointer of the tensor.
    /// \param tensor
    /// \return
    static T* AccessDataPointer(const tensorflow::Tensor &tensor) {
        // get underlying Eigen tensor
        auto tensor_map = tensor.tensor<T, NDIMS>();
        // get the underlying array
        auto array = tensor_map.data();
        return const_cast<T*>(array);
    }
};