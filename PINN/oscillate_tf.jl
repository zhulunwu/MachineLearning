using PyCall

tf=pyimport("tensorflow")
model(x) = tf.keras.layers.Dense(1,activation="sigmoid")(x)
y = tf.constant(1.0)
