using Flux: Dense, σ, Chain, softmax
using Plots
using Random
# using ScikitLearn
# using TensorFlow

#gonna use ScikitLearn here to split train and test sets (in windows)


#####################################
#EXAMPLE of 2 (linear) layers related by some non-linearity which may be represented by the sigmoid function.

# W1 = rand(3,5)
# println(W1)
# b1 = rand(3)
# println(b1)
# layer1(x) = W1*x .+ b1

# W2 = rand(2,3)
# println(W2)
# b2 = rand(2)
# println(b2)
# layer2(x) = W2*x .+ b2

# model(x) = layer2(σ.(layer1(x)))

# model(rand(5))

#####################################
#Here's a function which returns linear layers, to avoid repitition:
# function linear_layers(inputs,outputs)
#     W = randn(outputs, inputs)
#     b = randn(outputs)
#     x -> W*x .+ b
# end

# lin_layer1 = linear_layers(5,3)
# lin_layer2 = linear_layers(3,2)

# model(x) = lin_layer2(σ.(lin_layer1(x)))
# model(rand(5))

#####################################
# Or create a struct that explicitly represents the "affine" layer
# struct AffineLayer 
#     W
#     b
# end

# AffineLayer(inputs::Integer, outputs::Integer =
#     AffineLayer(randn(outputs,inputs),randn(outputs))

# #and create an overload call, so that the struct object can be used as a function:
# (m::AffineLayer)(x) = m.W * x .+ m.b

# a = AffineLayer(10, 5)

# a(rand(10)) # => 10-element input vector, 5-element output vector

#####################################
#AND THIS IS BASICALLY THE Flux.Dense() FUNCTION:
#(except that Dense also takes an activation function as input for convenience)

# layer1 = Dense(10,5,σ)
# layer2 = Dense(5,2,σ)

# model(x) = layer2(layer1(x))

# model(rand(10))

#####################################
#Flux also lets you create "chains" of layers! :D

model = Chain(
    Dense(10,5,σ),   #a dense layer with 10 inputs and 5 outputs
    Dense(5,3),      #5 inputs and 3 outputs
    softmax
    #The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1
)

# Random.seed!(1)
# rng = rand(10)
# println(rng)

# Random.seed!(1)
# rng2 = rand(10)
# println(rng2)

Random.seed!(1)
model(10)