using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics: mean
using Base.Iterators: partition
using Flux:Params,update!

include("data.jl")
include("model.jl")

# 模型
m=vgg16()
loss(x, y)=crossentropy(m(x), y)
opt = ADAM()

# 训练
n=0;showinfo()=begin global n;print("\r",n/500,"%已完成");n+=1;end

for i=1:5
  images_train,labels_train=train_image_label(i);
  data_train = [(cat(images_train[i]..., dims = 4), labels_train[:,i]) for i in partition(1:10000, 100)]
  Flux.train!(loss,params(m),data_train, opt,cb=showinfo)
end

# 测试
testimgs,lbls_bytes=test_image_label()
testX = cat(testimgs..., dims = 4)
testY = onehotbatch(lbls_bytes,0x00:0x09)
accuracy(x, y) = mean(onecold(m(x), 1:10) .== onecold(y, 1:10))
@show(accuracy(testX, testY))