using Flux, Flux.Data.MNIST, Statistics
using Flux: Tracker, throttle, params, binarycrossentropy, crossentropy
using Flux.Tracker: update!
using NNlib: relu, leakyrelu
using Base.Iterators: partition
using Images: channelview
using BSON: @save
using CuArrays
using CUDAnative:exp, log

include("data_loader.jl")
# imgs = MNIST.images()

BATCH_SIZE = 512
REAL_LABEL =  ones(1, BATCH_SIZE) |> gpu
FAKE_LABEL = zeros(1, BATCH_SIZE) |> gpu
#
# data = [reshape(hcat(Array(channelview.(imgs))...), 28, 28, 1,:) for imgs in partition(imgs, BATCH_SIZE)]
# data = gpu.(data)

train_data = load_dataset_as_batches("../celeba-dataset/img_align_celeba/img_align_celeba/", BATCH_SIZE)
train_data = gpu.(train_data)
println(size(train_data))

println(size(train_data))

NUM_EPOCHS = 50
training_steps = 0
GAMMA = Float32(25)
BETA = Float32(5)

discriminator_eta = 0.0001f0
generator_eta = 0.0001f0

encoder_features = Chain(Conv((5,5), 3 => 32,leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(32),
	Conv((5, 5), 32 => 64, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(64),
	Conv((5, 5), 64 => 128, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	Conv((5, 5), 128 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	x -> reshape(x, :, size(x, 4)),
	Dense(4096, 2048),
	x -> relu.(x)) |> gpu

encoder_mean = Chain(encoder_features, Dense(1024*2, 512)) |> gpu

encoder_logsigma = Chain(encoder_features, Dense(1024*2, 512), x -> tanh.(x)) |> gpu

encoder_latent(x) = encoder_mean(x) + randn(512, 512) * exp.(encoder_logsigma(x)./2) |> gpu

decoder_generator = Chain(Dense(512, 32*8*4*4),
	x -> reshape(x, 4, 4, 256, :),
	BatchNorm(256),
	x -> relu.(x),
	ConvTranspose((4, 4), 256 => 128, relu; stride = (2, 2), pad = (1, 1)),
	BatchNorm(128),
	ConvTranspose((4, 4), 128 => 64, relu; stride = (2, 2), pad = (1, 1)),
	BatchNorm(64),
	ConvTranspose((4, 4), 64 => 32, relu; stride = (2, 2), pad = (1, 1)),
	BatchNorm(32),
	ConvTranspose((4, 4), 32 => 3, relu, stride = (2, 2), pad = (1, 1)),
	x -> tanh.(x)) |> gpu

discriminator_similar = Chain(Conv((5, 5), 3 => 64, leakyrelu, stride = (2, 2), pad = (2, 2)),
	Conv((5, 5), 64 => 128, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(128),
	Conv((5, 5), 128 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	Conv((5, 5), 256 => 256, leakyrelu, stride = (2, 2), pad = (2, 2)),
	BatchNorm(256),
	x -> reshape(x, :, size(x, 4)),
	Dense(1024*4, 1)) |> gpu

discriminator = Chain(discriminator_similar, x -> sigmoid.(x)) |> gpu

function auxiliary_Z(latent_vector)
	return abs.(randn(size(latent_vector)...))
end

opt_encoder = ADAM(0.0003, (0.9, 0.999))
opt_decgen = ADAM(0.0003, (0.9, 0.999))
opt_discriminator = ADAM(0.0003, (0.9, 0.999))

function bce(ŷ, y)
    mean(-y.*log.(ŷ) - (1  .- y .+ 1f-5).*log.(1 .- ŷ .+ 1f-5))
end
#bce(y1, y) = -y * log(y1) - (1 - y)*log(1 - y1)

function prior_loss(latent_vector, auxiliary_Z)
	entropy = sum(latent_vector .* log.(latent_vector)) *1 //size(latent_vector,2)
 	cross_entropy = crossentropy(auxiliary_Z, latent_vector)
 	return entropy + cross_entropy
end

function discriminator_loss(X)
	latent_vector = encoder_latent(X)
	X_reconstructed = decoder_generator(latent_vector)
	Z_prior = auxiliary_Z(latent_vector) |> gpu
	X_p = decoder_generator(Z_prior)
	reconstruction_loss = bce(discriminator(X_reconstructed), FAKE_LABEL)
	sampling_loss = bce(discriminator(X_p), FAKE_LABEL)
	real_loss = bce(discriminator(X), REAL_LABEL)
	return mean(reconstruction_loss) + mean(sampling_loss) + mean(real_loss)
end

function decoder_loss(X)
	latent_vector = encoder_latent(X)
	X_reconstructed = decoder_generator(latent_vector)
	x_sim = discriminator_similar(X_reconstructed)
	x_sim_real = discriminator_similar(X)
	reconstruction_loss = Flux.mse(x_sim, x_sim_real)
	return GAMMA * reconstruction_loss - discriminator_loss(X)
end

function encoder_loss(X)
	latent_vector = encoder_latent(X)
	log_sigma = encoder_logsigma(X)
	enc_mean = encoder_mean(X)
	Z_prior = auxiliary_Z(latent_vector) |> gpu
	X_reconstructed = decoder_generator(latent_vector)
	x_sim = discriminator_similar(X_reconstructed)
	x_sim_real = discriminator_similar(X)
	reconstruction_loss = Flux.mse(x_sim, x_sim_real)
	return -0.5f0 * sum(1.0f0 .+ log_sigma .- (enc_mean .* enc_mean) .- exp.(log_sigma))/(BATCH_SIZE*784) + BETA * reconstruction_loss
end

function save_weights(enc, dec_gen, disc)
	enc = enc |> cpu
	dec_gen = dec_gen |> cpu
	disc = disc |> cpu
	@save "../weights/enc.bson" enc
	@save "../weights/dec_gen.bson" dec_gen
	@save "../weights/disc" disc
	enc = enc |> gpu
	dec_gen = dec_gen |> gpu
	disc = disc |> gpu
end

function training(X)
	println("starting...")
	gradient_dis = Flux.Tracker.gradient(() -> discriminator_loss(X), params(discriminator))
	println("G1")
	gradient_dec = Flux.Tracker.gradient(() -> decoder_loss(X), params(decoder_generator))
	println("G2")
	gradient_enc = Flux.Tracker.gradient(() -> encoder_loss(X), params(encoder_mean, encoder_logsigma))
	println("G3")
	update!(opt_discriminator, params(discriminator), gradient_dis)
	println("U1")
	update!(opt_decgen, params(decoder_generator), gradient_dec)
	prinltn("U2")
	update!(opt_encoder, params(encoder_mean, encoder_logsigma), gradient_enc)

	return discriminator_loss(X), decoder_loss(X), encoder_loss(X)
end
# println(size(train_data[1]))
# training(train_data[1])

SAVE_FREQ = 800
function train_all()
	i = 0
	for epoch in 1:NUM_EPOCHS
		println("-------- Epoch : $epoch ---------")
		for X in train_data
			dis_loss, dec_loss, enc_loss = training(X |> gpu)
			i += 1
			if i % SAVE_FREQ
				save_weights(encoder_mean, decoder_generator, discriminator)
			end
		end
	end
end

train_all()
