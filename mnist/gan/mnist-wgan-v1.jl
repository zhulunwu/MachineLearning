using Flux,GR,Statistics
using Distributions: Uniform
using Flux: back!, testmode!
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
c = 0.01f0
gen_update_frq = 5

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

opt=RMSProp(0.0001)

training_step = 0
#################
function nullify_grad!(p)
    if typeof(p) <: TrackedArray
      p.grad .= 0.0f0
    end
    return p
  end
  
function zero_grad!(model)
    model = mapleaves(nullify_grad!, model)
end

function updateparams(opt,ps)
    for x in ps
      Tracker.update!(opt, x, Tracker.grad(x))
      x.tracker.grad = Tracker.zero_grad!(x.tracker.grad)
    end
end

dist=Uniform(-1,1)
function train(x)
    global training_step
    z = rand(dist, z_dim, BATCH_SIZE) 
  
    zero_grad!(D) 

    D_loss=d_loss(x,z)
    Flux.back!(D_loss)
    updateparams(opt,params(D))
  
    for p in params(D)
      p.data .= clamp.(p.data, -c, c)
    end
   
    if (training_step+1) % gen_update_frq == 0
      zero_grad!(G)
      z = rand(dist, z_dim, BATCH_SIZE)    
      G_loss=g_loss(nothing,z)
      Flux.back!(G_loss)
      updateparams(opt,params(G))
  
      println("D loss: $(D_loss.data) | G loss: $(G_loss.data)")
    end
  
    training_step += 1
  end  
#################
img(x) = reshape((x.+1)/2, 28, 28)
function sample()
    # 16 random digits
    noise = [rand(dist, z_dim, 1) for i=1:16] 
   
    testmode!(G)
    fake_imgs = img.(map(x -> G(x).data, noise))
    testmode!(G, false)
  
    vcat([hcat(imgs...) for imgs in partition(fake_imgs, 4)]...)
  end


while training_step<10000
    imgs=onebatch(BATCH_SIZE)
    train(imgs)    

    # 显示产生的图像
    if training_step%200==0
        figs=sample()
        imshow(figs',colormap=2)
    end
end