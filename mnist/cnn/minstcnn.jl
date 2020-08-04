using Base.Iterators: repeated, partition
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Dates
using Makie
include("data.jl")
# 读入图像和标签
images = float.(minst_images()) 
imgs = [Float32.(images[i]) for i=1:length(images)] #以后的Flux可能会无需此代码。
lbls = onehotbatch(minst_labels(),0:9)

# 创建cnn神经网络，仅测试函数，不具备应用价值。
m = Chain(  Conv((3, 3), 1=>1, relu),
            MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(169, 10), softmax)

# 数据批量化处理
data_train = [(cat(imgs[i]..., dims = 4), lbls[:,i])
         for i in partition(1:length(imgs), 32)]

# 损失函数
function loss(input, label)
     l=crossentropy(m(input), label)
     println("loss=",l)
     return l
end

Flux.train!(loss, params(m), data_train,ADAM())