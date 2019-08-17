import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using TensorFlow, XLA, Flux, Printf

ENV["COLAB_TPU_ADDR"] = "10.240.1.2:8470"
sess = Session(Graph(); target="grpc://10.240.1.2:8470")
run(sess, TensorFlow.Ops.configure_distributed_tpu())

dx = 1.0
dy = 1.0
dz = 1.0

c1 = Float32.(1.0/24.0)
c2 = Float32.(-1.0/24.0)
c3 = Float32.(9.0/8.0)
c4 = Float32.(-9.0/8.0)

c1_ =  [c1/dx, c1/dy, c1/dz]
c2_ =  [c2/dx, c2/dy, c2/dz]
c3_ =  [c3/dx, c3/dy, c3/dz]
c4_ =  [c4/dx, c4/dy, c4/dz]

d1_ = 1
d2_ = 2
d3_ = 0
d4_ = 1

function apply(fijk1, fijk2, fijk3, fijk4, c1, c2, c3, c4)
        return  c1 * fijk1 + c2 * fijk2 + c3 * fijk3 + c4 * fijk4
end

d = 1
i = 5
j = 5

x = 50
y = 50
z = 50000

fijk = Float32.(reshape(1:(x*y*z), (x, y, z)))
fijk1 = fijk[i + d1_, j, :]
fijk2 = fijk[i - d2_, j, :]
fijk3 = fijk[i + d3_, j, :]
fijk4 = fijk[i - d4_, j, :]

@time compld = @tpu_compile apply(XRTArray(fijk1), XRTArray(fijk2), XRTArray(fijk3), XRTArray(fijk4), XRTArray(c1), XRTArray(c2), XRTArray(c3), XRTArray(c4))
@time run(compld, XRTArray(fijk1), XRTArray(fijk2), XRTArray(fijk3), XRTArray(fijk4), XRTArray(c1), XRTArray(c2), XRTArray(c3), XRTArray(c4))
