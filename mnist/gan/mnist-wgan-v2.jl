using Flux,GR,Statistics
using Distributions: Uniform
using Flux: testmode!
using Base.Iterators: partition

include("data.jl")

images = minst_images()

function onebatch(batchsize::Int)
    data=rand(images,batchsize)
    imgs=[Float64.(data[i]) for i=1:batchsize]
    imgs = hcat(reshape.(imgs, :)...)
    return imgs
end

# 一些参数
BATCH_SIZE = 64
X_dim = 784
z_dim = 100
h_dim = 128

D=Chain(Dense(X_dim,h_dim,relu),Dense(h_dim,1))
G=Chain(Dense(z_dim,h_dim,relu),Dense(h_dim,X_dim,sigmoid))

function d_loss(x,z)
    dl= mean(D(G(z)))-mean(D(x))
    return dl
end
function g_loss(nothing,z)
    gl  = -mean(D(G(z)))
    return gl
end

function nullify_grad!(p)
    if typeof(p) <: TrackedArray
      p.grad .= 0.0f0
    end
    return p
  end
  
function zero_grad!(model)
    model = mapleaves(nullify_grad!, model)
end

img(x) = reshape((x.+1)/2, 28, 28)
function sample()
    noise = [rand(Uniform(-1,1), z_dim, 1) for i=1:16] 
   
    testmode!(G)
    fake_imgs = img.(map(x -> G(x).data, noise))
    testmode!(G, false)
  
    vcat([hcat(imgs...) for imgs in partition(fake_imgs, 4)]...)
  end


for step=1:10000
    imgs=onebatch(BATCH_SIZE)
    z = rand(Uniform(-1,1), z_dim, BATCH_SIZE)
    zero_grad!(D) 
    Flux.train!(d_loss, params(D),[(imgs,z)],RMSProp(0.0001))
    for p in params(D)
        p.data .= clamp.(p.data, -0.01, 0.01)
      end
    zero_grad!(G)
    Flux.train!(g_loss, params(G),[(nothing,z)],RMSProp(0.0001))
    
    # 显示产生的图像
    if step%200==0
        figs=sample()
        imshow(figs',colormap=2)
        println("step=",step)
    end

end