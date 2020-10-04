#CNN --> Convolutional Neural Network

# collect sound samples:
    # footsteps
    # dog barking
    # gun shots
    # venster wat breek
    # car horn
    # alarms (house,car)
    # moving vehicle
    # voices
    # whistle

# perform pre-processing (make sample lengths equal, apply volume balancing)
    # use julia.images

# train CNN (using flux?)
# or use a pre-trained CNN (vgg net)

#program functionalities:
# show and add to log when too much noise/distortion to classify

using Flux: gradient, params
using Plots: plot, plot!

println("###################################################################################################################################################")

# each image is a 28×28 array of Gray colour values (see Colors.jl).
# Dataset:	Classes	traintensor	trainlabels	testtensor	testlabels
# MNIST: 	10	    28x28x60000	60000	    28x28x10000	10000
# img = Flux.Data.MNIST.images()
# println(size(img))
# println(sizeof(img))
# println(typeof(img))
# println(length(img))
# lab = Flux.Data.MNIST.labels()

# println(img[2])
# display(plot(img[5]))
# display(plot(plot(img[1]), plot(img[2]), plot(img[3]), plot(img[4])))

# println(lab[1])
# println(lab[2])
# println(lab[3])
# println(lab[4])


################################################ Julia 60-minute-blitz.jl ################################################

# x = rand(Float32,2,3)

# println(x)
# println(length(x))
# x = rand(BigFloat,2,3)
# println(x)
# println(length(x))

# x[2,3]
# println(x[2,3])
# plot(x)
# x[:,3]

# x+x
# x-x
# x

# zeros(5,5) .+ (1:5)
# (1:5) .* (1:5)'   #times table

###########################

#matrix multiplication:
# W = randn(Float32,5,10) #5 rows, 10 columns
# println(W)
# x = rand(Float32,10)
# println(x)
# W * x

###########################

#Automatic Differentiation
# f(x) = 3x^2 + 2x + 1
# f(2) # 17

# df(x) = gradient(f,x)[1]
# df(2) # 14

# ddf(x) = gradient(df,x)[1]
# ddf(2) # 6


# f(x,y) = sum((x .- y).^2)

# x = [2,1]
# y = [2,0]

# println(gradient(f,x,y)) # for ez derivatives (such as these)

# but for multiple parameters you take derivatives with 'params':
# grads = gradient(params(x,y)) do 
#     f(x,y)
# end

# display(grads[x])
# display(grads[y])
# println(grads[x])
# println(grads[y])


##########################
# SIMPLE LINEAR REGRESSION: (try to predict an output array y from an input array x.)

W = rand(2,5) #weights
println("W (weights):")
display(W)
b = rand(2) #bias
println("b (bias):")
display(b)

predict(x) = W*x .+ b

# Dummy data
x = rand(5) #input array (same dimension as weights, W, for matrix multiplication)
println("x (input array):")
display(x)
y = rand(2) #output array

function loss(x,y)
    println("(actual) y:")
    println(y)
    ŷ = W*x .+ b
    println("Predicted y:")
    println(ŷ)
    Error = sum((y .- ŷ).^2) #loss
    # println(typeof(Error))
    # println("Error:")
    # println(Error)
    # return Error, ŷ
end

# Error, ŷ = loss(x,y)
# println("Error:")
# println(Error)

#To improve the prediction we can take the gradients of W and b with respect to the loss and perform gradient descent:
# grads = gradient(() -> Error, params(W, b))
grads = gradient(() -> loss(x,y), params(W, b))

W̄ = grads[W]
display(W̄)

W .-= 0.1 .* W̄ #performing gradient descent

display(loss(x,y))
# Error, ŷ = loss(x,y)
# display(Error)

# plot(y,lab="actual")
# plot!(ŷ,lab="predicted")
# plot!(y .- ŷ,lab="error/loss/difference")

###########################
# Building Layers:

# W1 = rand(3, 5)
# b1 = rand(3)
# layer1(x) = W1 * x .+ b1

# W2 = rand(2, 3)
# b2 = rand(2)
# layer2(x) = W2 * x .+ b2

# model(x) = layer2(σ.(layer1(x)))

# model(rand(5)) # => 2-element vector

# #creating a function that returns linear layers:
# function linear(in, out)
#     W = randn(out, in)
#     b = randn(out)
#     x -> W * x .+ b
# end
# linear1 = linear(5, 3) # we can access linear1.W etc
# linear2 = linear(3, 2)

# model(x) = linear2(σ.(linear1(x)))

# model(rand(5))

###########################

# myloss(W, b, x) = sum(W * x .+ b)

# W = randn(3, 5)
# b = zeros(3)
# x = rand(5)

# a,b,c = gradient(myloss, W, b, x)
# println(a)
# println(b)
# println(c)

###########################

# W = randn(3,5)
# b = zeros(3)
# x = rand(5)

# println(W)
# println(summary(W))
# println(b)
# println(x)

# y(x) = sum(W * x .+ b)

# grads = gradient(()->y(x), params([W, b]))
# println(grads[W])
# println(grads[b])

###########################

# m = Dense(10,5)
# println(typeof(m))
# println(m)

# x = rand(Float32, 10)

# println(summary(params(m)))
# println(params(m))
# println(sizeof(params(m)))
# println(length(params(m)))

# params(m)

# m = Chain(Dense(10,5,relu),Dense(5,2),softmax)

# l(x) = sum(Flux.crossentropy(m(x),[0.5,0.5]))

# grads = gradient(params(m)) do
#     l(x)
# end

# for p in params(m)
#     println(grads[p])
# end

###########################

# `Training` a network reduces down to iterating on a dataset mulitple times, performing these
# steps in order.
# Just for a quick implementation, let’s train a network that learns to predict
# `0.5` for every input of 10 floats. `Flux` defines the `train!` function to do it for us.

# data = rand(10,100)
# labels = fill(0.5,2,100)

# m = Chain(Dense(10,5,relu),Dense(5,2),softmax)

# opt = Descent(0.01) # optimizer with learning rate ƞ (\nrleg)

# loss(x,y) = sum(Flux.crossentropy(m(x),y))
# Flux.train!(loss, params(m), [(data,labels)], opt)

###########################

# TRAINING A CLASSIFIER:



