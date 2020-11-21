using JLD: save, load
using MFCC
using Plots
using StatsPlots
using DataFrames: nrow
using WAV
using Flux
using RecursiveArrayTools
using BSON:  @save, @load

include("import.jl")
include("pre_processing.jl")

########################################################################################################################
#Import the UrbanSound8K dataset:
metadataDF, classes, path_to_wav_files, path_to_metadata = importUrbanSound8K()

test_fold = 1
trainingDF, testDF = splitTrainTest(metadataDF,test_fold)

#Get MFCC's (directly from wav files):
# generate_mfccs(trainingDF,testDF,path_to_wav_files) #only run if not saved previously

test_mfccs = load(string(path_to_wav_files,"test_mfccs.jld"), "mfccs") # array of test mfccs

training_mfccs = load(string(path_to_wav_files,"training_mfccs.jld"), "mfccs") # array of training mfccs
########################################################################################################################
#Perform Pre-Processing on the data:

index = 77 # index in test set to select (48 used in report)
some_MFCC, sizeOfInput = get_MFCC(index) # get the specified value from the test data set
println(sizeOfInput)


########################################################################################################################
#Build the model
# println("Flattening sample MFCC..")
# println(size(some_MFCC))
# some_MFCC = collect(Iterators.flatten(some_MFCC)) # * remember
# println("Flattened (input):")
# println(size(some_MFCC))

sizeOfOutput = 10

println(classes)
# classes = ["0 = air_conditioner", "1 = car_horn", "2 = children_playing", "3 = dog_bark","4 = drilling", "5 = engine_idling", "6 = gun_shot", "7 = jackhammer", "8 = siren", "9 = street_music"]

# correct_answer = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # the  correct_answer
# println(correct_answer)
correct_answer = Flux.onehot("dog_bark", classes)
# println(correct_answer)
# wag
# labels = fill(correct_answer, 500)
labels = [correct_answer]
# labels= Flux.onehotbatch(classes, classes)
# println(labels)
# wag
# println(size(labels))
# vector_of_array = RecursiveArrayTools.VectorOfArray(labels)
# labels = convert(Array,vector_of_array)
# labels = labels'

x = training_mfccs.values#[30:100]
keepers = []
for mfcc in x
    if size(mfcc) == (398, 13)
        mfcc = collect(Iterators.flatten(mfcc))
        push!(keepers,mfcc)
    end
    # println(size(keepers))
    if size(keepers)[1] == 200 #let's consider all the samples
        break
    end
end
data = keepers
# for mfcc in x
#     println(size(mfcc))
# end
# println(size(data))
# vector_of_array = RecursiveArrayTools.VectorOfArray(data)
# data = convert(Array,vector_of_array)
# data = data'
# println(size(data))
# wag
# Manet 'n gewone MLP: (try CNN's definitief ook)
model = Chain(
    # x -> reshape(x, :, size(x)[end]),
    # x -> Flux.flatten,
    Dense(sizeOfInput,128,relu), #input layer
    # Dropout(0.25),
    # Dense(512,128,sigmoid), #1st hidden layer
    Dense(128,32,sigmoid), #2nd hidden layer (making it DEEP learning...)
    # Dropout(0.5),
    Dense(32,sizeOfOutput), #output layer
    softmax
    #test
    # The softmax function is a function that turns a vector of K real values into a vector of K real values THAT SUM TO 1
) |> Flux.gpu

# params = Flux.params(model)
# println(params) #don't do it..

# prediction = model(some_MFCC)
# println(prediction)
# wag

# loss(x,y) = Flux.Losses.mse(model(x),y) #mean-squared error/loss
# loss(prediction,correct_answer) = sum(Flux.crossentropy(prediction,correct_answer)
loss(x, y) = sum(Flux.crossentropy(model(x), y))

# df = Flux.gradient(f,x)[1]
# gradients = Flux.gradient(Flux.params(model)) do loss(x, y) end

# for p in params[1]
#     println(p) #DON'T DO IT..
# end

ƞ = 0.1 #learning rate
optimiser = Flux.Optimise.Descent(ƞ)

# x = rand(5)
# y = rand(2)

# println(size(x))
# println(size(y))
# wag

# data = zip(x, y)
# data = Flux.Data.DataLoader(x, y, batchsize=3, partial=false)

# Flux.Optimise.train!(loss, params, data, optimiser)
println(loss)
# println(params)
println(size(data))
println(size(data[120]))
# println(labels[23])
# println(optimiser)

# data = [randn(5174), randn(5174), randn(5174)]
# labels = [randn(10), randn(10)]

# println(size(data))
# println(size(labels))

dataZIP = zip(data,labels)

# test_input = collect(Iterators.flatten(test_mfccs.values[55]))
# println(size(test_mfccs.values[55]))
# wag
i = rand(1:95)
println(i)
for mfcc in test_mfccs.values[i:i+5]
    if size(mfcc) == (398,13)
        global test_input = collect(Iterators.flatten(mfcc))
        # global test_input = mfcc
        break
    end
end
println("HIER")

another_test_input = test_mfccs.values[55]
j = rand(1:95)
println(j)
for mfcc in test_mfccs.values[j:j+5]
    if (size(mfcc) == (398,13)) && (mfcc != test_input)
        global another_test_input = collect(Iterators.flatten(mfcc))
        break
    end
end
println("HIER OOK")

println("hey")
println(test_input[1:10])
println(another_test_input[1:10])
# wag

# training_input = training_mfccs.values[55]
k = rand(1:95)
println(k)
for mfcc in training_mfccs.values[k:k+5]
    if size(mfcc) == (398,13)
        global training_input = collect(Iterators.flatten(mfcc))
        break
    end
end

# model = load("model.jld", "SimpleModel2")
# @load "mySimpleModel2.bson" model
# println(model)
# println(sizeof(Flux.params(model)))
# @load "myWeights.bson" weights
# Flux.loadparams!(model,weights)

# wag

predictionBefore = model(test_input)
evalcb() = @show(loss(test_input, correct_answer))
global train_accuracy = []

Flux.@epochs 25 Flux.Optimise.train!(loss, Flux.params(model), dataZIP, optimiser,
# cb = Flux.throttle(() -> println("training"), 2)
# cb = Flux.throttle(evalcb, 5)
cb = function ()
    # accuracy() > 0.9 && Flux.stop()
    prediction = model(test_input)
    println(prediction)
    # accuracy = @show(1-loss(prediction, correct_answer[1]))
    accuracy = @show(1-loss(test_input, correct_answer))
    push!(train_accuracy, accuracy)
  end
)

# weights = Flux.params(model)
# @save "myWeights.bson" weights

prediction = model(test_input)
println(string(predictionBefore, " (BEFORE TRAINING)"))
println(string(prediction, " (AFTER TRAINING)"))
another_prediction = model(another_test_input)
println(string("Test MFCC used lies between ", i, " and ", i+5))
println(string(another_prediction), " (ANOTHER PREDICTION)")
trained_prediction = model(training_input)
println(string(trained_prediction), " (TRAINED PREDICTION)")

global test_accuracy = []
for test_input in test_mfccs.values
    test_input = collect(Iterators.flatten(test_input))
    if size(test_input) == (398, 13)
        accuracy = 1 - loss(test_input, correct_answer)
        push!(accuracy, test_accuracy)
    end
end
# println(test_accuracy)

print("First prediction: ")
println(Flux.onecold(prediction, classes))
print("Another prediction: ")
println(Flux.onecold(another_prediction, classes))
print("TRAINED PREDICTION: ")
println(Flux.onecold(trained_prediction, classes))

display(Plots.bar(classes, predictionBefore, title="BEFORE TRAINING", ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.bar(classes, prediction, title="TEST 1 (AFTER TRAINING)", ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.bar(classes, another_prediction, title="TEST 2 (AFTER TRAINING)", ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.bar(classes, trained_prediction, title="TRAINED PREDICTION", ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.plot(train_accuracy, title="TRAINING ACCURACY", ylims=(0,1), xrotation=45, legend=:topright))
# display(Plots.plot(test_accuracy, title="TEST ACCURACY", ylims=(0,1), xrotation=45, legend=:topright))


# println("End of Program")
