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

index = 77 # index in test set to select (48 used in report)
some_MFCC, sizeOfInput = get_MFCC(index) # get the specified value from the test data set
println(sizeOfInput)

println("Flattening sample MFCC..")
println(size(some_MFCC))
some_MFCC = collect(Iterators.flatten(some_MFCC)) # * remember
println("Flattened (input):")
println(size(some_MFCC))

sizeOfOutput = 10

model = Chain(
    Dense(sizeOfInput,512,relu), #input layer
    Dense(512,128,relu), #1st hidden layer
    Dense(128,32,relu), #2nd hidden layer (making it deep learning...)
    Dense(32,sizeOfOutput), #output layer
    softmax
    # The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1
)

# loss(x,y) = Flux.Losses.mse(m(x),y)

# params = Flux.params(model)

model(some_MFCC)

# println("End of Program")
