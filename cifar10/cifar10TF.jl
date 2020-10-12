using PyCall
import Flux.onecold

include("data.jl")

tf=pyimport("tensorflow")
layers=tf.keras.layers

# model
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=true),metrics=["accuracy"])

# data
adddims(x)=reshape(x,(1,size(x)...))

for i=1:5
    global model
    images_train,labels_train=train_image_label(i)
    train_images=vcat(adddims.(images_train)...)
    train_labels=onecold(labels_train,Int8.(0:9))
    model.fit(train_images,train_labels, epochs=10)
end

images_test, labels_test=test_image_label()
test_images=vcat(adddims.(images_test)...)
test_labels=Int8.(labels_test)
model.evaluate(test_images,test_labels)


