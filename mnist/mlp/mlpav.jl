#mlp another version
using Flux
using Flux: onehotbatch,crossentropy,chunk,batch,throttle,onecold
using Flux: @epochs
using Statistics
import Flux.Data.DataLoader

include("data.jl")

images = Float32.(hcat(vec.(minst_images())...))
labels = onehotbatch(minst_labels(),0:9)

m=Chain(Dense(28^2,128,relu),Dropout(0.2),Dense(128,10),softmax)
loss(x,y)=crossentropy(m(x),y)
showinfo()=print("\rloss=",loss(images[:,1:10],labels[:,1:10]))

# dataloader方式加载数据
data_train=DataLoader((images,labels),batchsize=32)
@epochs 5 Flux.train!(loss, params(m),data_train,ADAM(),cb=showinfo)

#识别率
function Accuracy(model)
    image_test = map(x->Float32.(x),vec.(minst_images(:test)))
    label_test = minst_labels(:test)
    predicts=onecold.(m.(image_test),[0:9])
    return mean(label_test .== predicts)
end
println("识别率=",Accuracy(m))