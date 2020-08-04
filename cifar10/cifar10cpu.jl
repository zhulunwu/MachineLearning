using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics: mean
using Base.Iterators: partition
using Flux:Params,update!

include("data.jl")
include("model.jl")

# 先暂时考虑10000个文件的训练
images_train,labels_train=train_image_label(1);
data_train = [(cat(images_train[i]..., dims = 4), labels_train[:,i]) for i in partition(1:10000, 100)]

# 模型
m=vgg16()
loss(x, y)=crossentropy(m(x), y)
opt = ADAM()

# 训练
function train(loss, ps, data, opt)
    ps = Params(ps)
    for d in data
      gs = gradient(ps) do
        training_loss = loss(d...)
        println("loss=",training_loss)
        return training_loss
      end
      update!(opt, ps, gs)
    end
end
train(loss,params(m),data_train,opt)

# Flux.train!(loss,params(m),data_train, opt,cb=()->println("time=",now()))

# 测试
testimgs,lbls_bytes=test_image_label()
testX = cat(testimgs..., dims = 4)
testY = onehotbatch(lbls_bytes,0x00:0x09)

# Print the final accuracy
accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))
@show(accuracy(testX, testY))