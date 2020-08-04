# 逐层分析
using Flux
using Makie
using Images
using Base.Iterators

include("data.jl")

# 显示图像,一次显示一个批次,每批次应该N=n*n
function batshow(data)
    w,h,c,n=size(data)
    whc2cwh(d)=collect(colorview(RGB,permutedims(d,(3,1,2))))
    imgs=[image(rotr90(whc2cwh(data[:,:,:,i])),show_axis=false) for i=1:n];
    nr=Int(sqrt(n))
    rows=[hbox(p...) for p in partition(imgs,nr)];
    return vbox(rows...)
end

function img_channel(data)
    w,h,c,n=size(data) 
    data=data[:,:,:,1]  
    imgs=[image(data[:,:,i],show_axis=false) for i=1:c]
    r=Int(sqrt(c))
    rows=[hbox(p...) for p in partition(imgs,r)];
    return vbox(rows...)
end

function oneimg(data)
    w,h,c,n=size(data)
    whc=data[:,:,:,1]
    cwh=permutedims(whc,(3,1,2))
    return image(rotr90(collect(colorview(RGB,cwh))),show_axis=false)
end

# 读入数据
images_train,labels_train=train_image_label(1);
img_train = [cat(images_train[i]..., dims = 4) for i in partition(1:10000,1)];

imgdata=img_train[70]
oneimg(imgdata)
m=Conv((3, 3), 3 => 1)
out=m(imgdata)
image(out[:,:,1,1])

# 图像的池化
m=MaxPool((2,2))
out=m(imgdata)
oneimg(out)