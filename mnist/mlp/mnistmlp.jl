using Flux
using Flux: onehotbatch, crossentropy

include("data.jl")
# 读入图像和标签
images = minst_images()
labels = minst_labels()
dataset=zip(images,labels)

function nextbatch(batchsize::Int)
    data=rand(collect(dataset),batchsize)
    imgs=[Float32.(data[i][1]) for i=1:batchsize]
    imgs = hcat(reshape.(imgs, :)...)
    lbls=[data[i][2] for i=1:batchsize]
    lbls=onehotbatch(lbls,0:9)
    return imgs,lbls
end

m=Chain(Dense(28^2,128,relu),Dense(128,10),softmax)

function loss(x,y)
    l=crossentropy(m(x),y)
    println("loss=",l)
    return l
end

# 训练
for step=1:1000
    imgs,lbls=nextbatch(32)
    Flux.train!(loss, params(m),[(imgs,lbls)],RMSProp(0.0001))
end