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
n=1;showinfo()=begin global n;print("\r",n/500,"%已完成");n+=1;end

for i=1:5
  images_train,labels_train=train_image_label(i);
  labels_train=onehotbatch(labels_train,0:9)
  data_train = [(cat(images_train[i]..., dims = 4), labels_train[:,i]) for i in partition(1:10000, 100)]
  Flux.train!(loss,params(m),data_train, opt,cb=showinfo)
end

# 测试
testimgs,lbls_bytes=test_image_label();
sum=0
for i=1:length(testimgs)
    global sum
    img=reshape(testimgs[i],(size(testimgs[i])...,1))
    predict=onecold(m(img),0:9)
    predict == lbls_bytes[i] && (sum+=1)
    println("准确率=",sum/i)
end