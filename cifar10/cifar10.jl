using Flux, Statistics
using Flux: onehotbatch, onecold, crossentropy,batch,@epochs
import Flux.Data.DataLoader

include("data.jl")
include("model.jl")

# 模型
m=model() |>gpu
loss(x, y)=crossentropy(m(x), y)
opt = ADAM()

# 训练
n=1;
showinfo()=begin global n;print("\r$n/500");n+=1;end
@epochs 5 for i=1:5
  images_train,labels_train=train_image_label(i);
  images_train=batch(images_train) |> gpu
  labels_train=onehotbatch(labels_train,0:9) |> gpu
  data_train=DataLoader((images_train,labels_train),batchsize=100) 
  Flux.train!(loss,params(m),data_train, opt,cb=showinfo)
end

# 测试
images_test,labels_test=test_image_label()
images_test=batch(images_test)
data_test=DataLoader((images_test,labels_test),batchsize=100)
good=0
mdl=cpu(m)
for (x,y) in data_test
    global good
    good+=sum(onecold(mdl(x),0:9) .== y)
    println(good/length(labels_test))
end
println("准确率=",good/length(labels_test))