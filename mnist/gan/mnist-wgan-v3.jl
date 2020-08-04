using Flux, Statistics
using Flux: back!, testmode!
using Base.Iterators: partition
using NNlib: relu, leakyrelu
using Distributions: Uniform
using CUDAnative:tanh
using CuArrays
using GR

include("data.jl")
imgs = minst_images()

BATCH_SIZE = 64
training_step = 0
c = 0.01f0
gen_update_frq = 5 

data = [float(hcat(vec.(imgs)...)) for imgs in partition(imgs, BATCH_SIZE)]

NUM_EPOCHS = 1
noise_dim = 100
channels = 128
hidden_dim = 7 * 7 * channels

dist = Uniform(-1, 1)

# 生成器
fc_gen = Chain(Dense(noise_dim, 512), BatchNorm(512, relu),
            Dense(512, hidden_dim), BatchNorm(hidden_dim, relu))
deconv_ = Chain(ConvTranspose((4,4), channels=>64;stride=(2,2),pad=(1,1)), BatchNorm(64, relu),
                ConvTranspose((4,4), 64=>1, tanh;stride=(2,2), pad=(1,1)))

generator = Chain(fc_gen..., x -> reshape(x, 7, 7, channels, :), deconv_...) |> gpu

# 鉴别器
fc_disc = Chain(Dense(hidden_dim, 512), BatchNorm(512), 
					 x->leakyrelu.(x, 0.2f0), Dense(512, 1))
conv_ = Chain(Conv((4,4), 1=>64;stride=(2,2), pad=(1,1)), x->leakyrelu.(x, 0.2f0),
             Conv((4,4), 64=>channels; stride=(2,2), pad=(1,1)), BatchNorm(channels), 
			 x->leakyrelu.(x, 0.2f0))

discriminator = Chain(conv_..., x->reshape(x, hidden_dim, :), fc_disc...) |> gpu

opt_gen  = ADAM(0.001f0,(0.5f0,0.999f0))
opt_disc = ADAM(0.001f0,(0.5f0,0.999f0))

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
  # 16 random digits
  noise = [rand(dist, noise_dim, 1) |>gpu for i=1:16] 
 
  testmode!(generator)
  fake_imgs = img.(map(x -> cpu(generator(x).data), noise))
  testmode!(generator, false)

  vcat([hcat(imgs...) for imgs in partition(fake_imgs, 4)]...)
end

function updateparams(opt,ps)
  for x in ps
    Tracker.update!(opt, x, Tracker.grad(x))
    x.tracker.grad = Tracker.zero_grad!(x.tracker.grad)
  end
end

function train(x)
  global training_step
  z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
  inp = reshape(2x .- 1, 28, 28, 1, :) |> gpu

  zero_grad!(discriminator)
 
  D_real = discriminator(inp)
  D_real_loss = -mean(D_real)

  fake_x = generator(z)
  D_fake = discriminator(fake_x)
  D_fake_loss = mean(D_fake)

  D_loss = D_real_loss + D_fake_loss

  Flux.back!(D_loss)
  updateparams(opt_disc,params(discriminator))

  for p in params(discriminator)
    p.data .= clamp.(p.data, -c, c)
  end
 
  if (training_step+1) % gen_update_frq == 0
    zero_grad!(generator)
    z = rand(dist, noise_dim, BATCH_SIZE)  |> gpu
    fake_x = generator(z)
    D_fake = discriminator(fake_x)
    G_loss = -mean(D_fake)
    Flux.back!(G_loss)
    updateparams(opt_gen,params(generator))

    println("D loss: $(D_loss.data) | G loss: $(G_loss.data)")
  end

  training_step += 1
end

for e = 1:NUM_EPOCHS
  for imgs in data
    train(imgs)
  end
  println("Epoch $e over.")
end

imgdata=sample()
imshow(imgdata',colormap=2)