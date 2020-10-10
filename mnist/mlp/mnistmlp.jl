using Flux
using Flux: onehot,crossentropy,chunk,batch,throttle
using Flux: @epochs
using Statistics

include("data.jl")

# 图像转一维向量
function img2vec(image)
    f=map(Float32,image)
    return vcat(f...) 
end

# 读入图像和标签
images = img2vec.(minst_images())
labels = map(x->onehot(x,0:9),minst_labels())

nparts=6000
images_batch=chunk(images,nparts)
labels_batch=chunk(labels,nparts)
images_matrix=batch.(images_batch)
labels_matrix=batch.(labels_batch)

dataset=zip(images_matrix,labels_matrix)

m=Chain(Dense(28^2,64,relu),Dense(64,10),softmax)

loss(x,y)=crossentropy(m(x),y)

function showloss()
    selection=rand(1:60000,10)
    losssum=0
    for s in selection
        losssum+=loss(images[s],labels[s])
    end
    lossaverage=losssum*0.1
    print("\raverage loss=",lossaverage)
end

@epochs 5 Flux.train!(loss, params(m),dataset,RMSProp(0.0001),cb=throttle(showloss,1))

# 模型的使用
image_test = img2vec.(minst_images(:test))
label_test = minst_labels(:test)
predicts=argmax.(m.(image_test)).-1
mean(label_test .== predicts)
