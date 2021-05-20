using Flux

model()=Chain(
    Conv((3,3),3=>32,relu),
    MaxPool((2, 2)),
    Conv((3,3),32=>64,relu),
    MaxPool((2, 2)),
    Conv((3,3),64=>64,relu),
    x -> reshape(x, :, size(x, 4)),
    Dense(1024, 64, relu),
    Dense(64,10),
    softmax
)