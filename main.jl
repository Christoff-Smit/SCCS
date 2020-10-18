using JLD: save, load
using MFCC
using Plots
using DataFrames: nrow
using WAV
using Flux

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

index = 48 # index in test set to look at
show_MFCC(index)
this_MFCC = test_mfccs.values[index]
println(size(this_MFCC))
this_Path = test_mfccs.keys[index]
println(this_Path)

println(size(this_MFCC))
this_MFCC = collect(Iterators.flatten(this_MFCC)) #* remember
println(size(this_MFCC))

model = Chain(
    Dense(5174,512,relu), #input layer
    Dense(512,128,relu), #1st hidden layer
    Dense(128,32,relu), #2nd hidden layer
    Dense(32,10), #output layer
    softmax
    # The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1
)

# loss(x,y) = Flux.Losses.mse(m(x),y)

# params = Flux.params(model)

model(this_MFCC)

# println("End of Program")
