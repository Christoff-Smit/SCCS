using JLD: save, load
using MFCC
using Plots
using StatsPlots
using DataFrames: nrow, DataFrame
using WAV
using Flux: onecold, onehotbatch, Chain, Dense, Conv, relu, softmax, gpu, onehot, normalise, gradient, params
using Flux
using RecursiveArrayTools
using BSON:  @save, @load
using CUDAapi
using MLDatasets
using RandomNumbers
using Random
# if has_cuda()		# Check if CUDA is available
#     @info "CUDA is on"
#     import CuArrays		# If CUDA is available, import CuArrays
#     CuArrays.allowscalar(false)
# end

include("import.jl")
include("pre_processing.jl")

##################################################################
#Import the UrbanSound8K dataset:
meta_DF, classes, path_to_wav_files, path_to_metadata = importUrbanSound8K()



# # test_dict = Dict("x" => 10, "y" => 11, "z" => 12)
# test_dict = Dict()
# push!(test_dict, "x" => 20)
# push!(test_dict, "y" => 22)
# push!(test_dict, "z" => 24)
# println(test_dict)
# println(test_dict.keys)
# println(test_dict["x"])
# println(test_dict["y"])
# println(test_dict["z"])
# # println(test_dict.values) #doesn't work
# # println(size(test_dict)) #doesn't work
# save(string(path_to_wav_files,"test_dict.jld"), test_dict)
# loaded_dict = load(string(path_to_wav_files,"test_dict.jld"))
# println(typeof(loaded_dict))
# println(pairs(test_dict))
# println(length(pairs(test_dict)))
# # println(loaded_dict)
# println(loaded_dict["x"])
# println(loaded_dict["y"])
# println(loaded_dict["z"])

# wag ############## SUCCESS

##################################################################
# Generate MFCC's (directly from wav files):
# generate_mfccs(meta_trainingDF, meta_testDF, path_to_wav_files) #only run if not saved previously

# @time begin
# generate_mfccs(meta_DF, path_to_wav_files) #only run if not saved previously
# end
# wag
##################################################################
# Load previously generated MFCC's:
# test_mfccs = load(string(path_to_wav_files,"test_mfccs_dog_bark.jld"))#["mfccs"] # array of test mfccs
# train_mfccs = load(string(path_to_wav_files,"training_mfccs_dog_bark.jld"))#["mfccs"] # array of training mfccs

# test_mfccs = load(string(path_to_wav_files,"test_mfccs.jld") # array of test mfccs
# train_mfccs = load(string(path_to_wav_files,"train_mfccs.jld") # array of training mfccs

println("Loading pre-generated MFCC's..")
# @time begin
# MFCCs = load(string(path_to_wav_files,"MFCCs.jld"))
# MFCCs = load(string(path_to_wav_files,"MFCCs_small.jld"))
# MFCCs = load(string(path_to_wav_files,"MFCCs_small_augmented.jld"))
# MFCCs = load(string(path_to_wav_files,"MFCCs_augmented_numcep_50.jld"))
# MFCCs = load(string(path_to_wav_files,"MFCCs_augmented_numcep_13.jld")) #16 mins
# nrOfMFCCs = length(MFCCs)-5
# end

# println(size(test_mfccs.values))
# println(size(test_mfccs.keys)) #weird
# println(test_mfccs.keys) #weird
# println(pairs(MFCCs))
println(string(length(pairs(MFCCs))-5, " MFCCs (key/value pairs) generated"))
# println(MFCCs.keys[1:10])
# println(length(MFCCs.keys))
# println(MFCCs["101729-0-0-36.wav"])
# wag

fourSecond_metaDF = DataFrame()
unsupported_compression_code_indices = [4804,6247,6248,6249,6250,6251,6252,6253,8339]
# unsupported_compression_code_indices = [608]
# augmented_unsupported_compression_code_indices = [(608-1)*4,(608-1)*4+1,(608-1)*4+2,(608-1)*4+3] #shifted for augmented data
augmented_unsupported_compression_code_indices = []
for i in unsupported_compression_code_indices
    push!(augmented_unsupported_compression_code_indices, (i-1)*5+1)
    push!(augmented_unsupported_compression_code_indices, (i-1)*5+2)
    push!(augmented_unsupported_compression_code_indices, (i-1)*5+3)
    push!(augmented_unsupported_compression_code_indices, (i-1)*5+4)
    push!(augmented_unsupported_compression_code_indices, (i-1)*5+5)
end
println(augmented_unsupported_compression_code_indices)
# wag

global indexCounter = 0
global nr_of_notFourSeconds = 0
global nr_of_unsupported = 0
for row in eachrow(meta_DF)#[8732-1000:8732]
    global indexCounter += 1
    # println(row)
    # println(row."end"-row.start)
    if (row."end"-row.start == 4)
        global nr_of_notFourSeconds += 1
    else
        if !(indexCounter in unsupported_compression_code_indices)
            global nr_of_unsupported += 1
        else
            push!(fourSecond_metaDF, row)
        end
    end
end
println(string(nr_of_notFourSeconds, " samples removed because not 4 seconds long"))
println(string(nr_of_unsupported, " samples removed because unsupported compression code"))
# println(fourSecond_metaDF)
println(size(fourSecond_metaDF))
println(nrow(fourSecond_metaDF))
wag
percentageLoss = (1-size(fourSecond_metaDF)[1]/size(meta_DF)[1])*100 # the % of all samples lost by only considering those that are 4 seconds long
println(string(percentageLoss, " % of samples discarded (4 second condition + unsupported compression code error)"))
print(size(fourSecond_metaDF))
print(" out of ")
print(size(meta_DF))
println(" remaining")
wag

##################################################################
#Split train test data sets
test_fold = 1 # fold (out of 1->10) to be used as the test set in this iteration

test_mfccs = []
train_mfccs = []

test_metaDF = DataFrame()
train_metaDF = DataFrame()

println("Splitting training and test data & labels:")
println(string("(Test fold = ", test_fold, ")"))

for row in eachrow(fourSecond_metaDF)
    # println(typeof(MFCCs))
    mfcc = MFCCs[row.slice_file_name]
    shifted_left_2 = MFCCs[string(row.slice_file_name[1:length(row.slice_file_name)-4],"_shift_left_2.wav")]
    shifted_left_1 = MFCCs[string(row.slice_file_name[1:length(row.slice_file_name)-4],"_shift_left_1.wav")]
    shifted_right_1 = MFCCs[string(row.slice_file_name[1:length(row.slice_file_name)-4],"_shift_right_1.wav")]
    shifted_right_2 = MFCCs[string(row.slice_file_name[1:length(row.slice_file_name)-4],"_shift_right_2.wav")]
    # println("Hey")
    # wag
    if row.fold == test_fold
        push!(test_mfccs, mfcc') #just push the original mfcc for testing purposes
        push!(test_metaDF, row)
    else
        push!(train_mfccs, mfcc')
        push!(train_metaDF, row)
        
        push!(train_mfccs, shifted_left_2')
        push!(train_metaDF, row)
        
        push!(train_mfccs, shifted_left_1')
        push!(train_metaDF, row)
        
        push!(train_mfccs, shifted_right_1')
        push!(train_metaDF, row)
        
        push!(train_mfccs, shifted_right_2')
        push!(train_metaDF, row)
    end
end

println("Training data:")
println(size(train_mfccs))
println(size(train_metaDF))
# println(first(train_metaDF, 5))
# println(last(train_metaDF, 5))
# println(size(1:nrow(train_metaDF)))
# println(shuffle(1:nrow(train_metaDF)))
# wag
shuffled_indices = shuffle(1:nrow(train_metaDF))
# println(shuffled_indices[1:5])
train_mfccs = train_mfccs[shuffled_indices]
train_metaDF = train_metaDF[shuffled_indices, :]
# println(first(train_metaDF, 5))
# println(last(train_metaDF, 5))
# println(shuffled_indices[1:5])
# wag

println("Test data:")
println(size(test_mfccs))
println(size(test_metaDF))
wag



##################################################################
#Perform Pre-Processing on the data:
#e.g. data augmentation


##################################################################
#Do some investigation on a specific sample:

# index = rand(1:size(test_metaDF)[1])
index = 10
global slice_file_name = train_metaDF.slice_file_name[index]
global fold = train_metaDF.fold[index]
global class = train_metaDF.class[index]
# okay = 0
# while okay == 0
#     index = rand(1:size(fourSecond_metaDF)[1])
#     println(index)
#     # (dog test set index 48 used in report)
#     global slice_file_name = fourSecond_metaDF.slice_file_name[index]
#     global fold = fourSecond_metaDF.fold[index]
#     global class = fourSecond_metaDF.class[index]
#     println(class)
#     if (class != "engine_idling" && class != "drilling" && class != "jackhammer" && class != "air_conditioner")
#         global okay = 1
#         println("DONE")
#     end
# end

# random_MFCC, sizeOfInput = get_one_MFCC(slice_file_name, fold, MFCCs) # get the specified value from the dictionary of MFCC's
# sizeOfInput = 398*13 # = 5174
sizeOfInput = 100*13 # 1300
println(sizeOfInput)
println(class)
# wag

##################################################################
#Build the model

println(classes)
nclasses = length(classes)

# sizeOfInput = 748
# Manet 'n gewone MLP: (try CNN's definitief ook)
# model = Chain(
#     # x -> reshape(x, :, size(x)[end]),
#     # x -> Flux.flatten,
#     Dense(sizeOfInput,128,relu), #input layer
#     # Dense(512,128,relu), #1st hidden layer
#     Dropout(0.5),
#     Dense(128,32,relu), #2nd hidden layer (making it DEEP learning...)
#     Dropout(0.5),
#     Dense(32,nclasses), #output layer (nclasses = 10)
#     # softmax #LEAVE OUT !!!
#     #test
#     # The softmax function is a function that turns a vector of K real values into a vector of K real values THAT SUM TO 1
# ) |> gpu

# CONV:
model = Chain(
    Conv((5,5), 1 => 24, relu), # (96,9)
    MaxPool((4,2)), # (24,5)

    # Conv((1,1), 24 => 48, relu), # (24,5)
    
    Conv((3,3), 24 => 48, relu), # (22,3)
    MaxPool((4,2)), # (7,3)
    
    Conv((3,3), 48 => 48, relu), # (5,5)
    # MaxPool((4,1)), # (5,5)
    # AdaptiveMaxPool((2,2)),

    flatten,
    Dropout(0.5),
    # Dense(1008,512,relu),
    # Dropout(0.5),
    # Dense(512,128,relu),
    # Dropout(0.5),
    # Dense(128,64,relu),
    # Dropout(0.5),
    Dense(64,32,relu),
    Dropout(0.5),
    Dense(32,10),
    softmax
) |> gpu

# println(Flux.outdims(model, (28,28)))
println(Flux.outdims(model, (100,13)))
wag


# println(size(x))
# println(size(y))
# wag

# data = zip(x, y)

# Flux.Optimise.train!(loss, params, data, optimiser)


# Get data:
# println("Flattening sample MFCC..")
# println(size(random_MFCC))
# random_MFCC = collect(Iterators.flatten(random_MFCC)) # * remember
# println("Flattened (input):")
# println(size(random_MFCC))

# println(train_mfccs)
println(size(train_mfccs))
# println(train_mfccs[1])
println(size(train_mfccs[1]))
# wag

flattened_train_mfccs = []
global counter = 0
global oneSecondCounter = 0
global train_sizeCheckCounter = 0
global isnan_counter = 0
global train_isnan_indices = []
global train_incorrect_size_indices = []
global train_indices = []
for mfcc in train_mfccs
    global counter += 1
    # println(size(mfcc))
    # wag
    if size(mfcc) == (398, 13)
        # mfcc = collect(Iterators.flatten(mfcc))
        mfcc = Flux.normalise(mfcc, dims=1) #normalise
        # println(mfcc[5170:5174])
        # wag
        # println(size(mfcc))
        my_zero_padding = zeros(2,13)
        # println(my_zero_padding)
        padded_mfcc = vcat(mfcc, my_zero_padding) #apply zero padding
        # println(mfcc[395:398,:])
        # println(padded_mfcc[395:400,:])
        # splitting each mfcc in 4:
        for i in 1:4
            global oneSecondCounter += 1
            this_oneSecond_slice = padded_mfcc[(1+(i-1)*100:i*100),:]
            # println("####################################")
            # println(size(this_oneSecond_slice))
            # println(typeof(this_oneSecond_slice))
            this_oneSecond_slice = reshape(this_oneSecond_slice, (size(this_oneSecond_slice)...,1))
            this_oneSecond_slice = reshape(this_oneSecond_slice, (size(this_oneSecond_slice)...,1))
            # println(size(this_oneSecond_slice))
            # println(model(this_oneSecond_slice))
            # println("HIER")
            # wag
            # flat_oneSecond_slice = collect(Iterators.flatten(this_oneSecond_slice)) #CONV
            # println(size(flat_oneSecond_slice))
            # wag
            # if !any(isnan, model(flat_oneSecond_slice))
            if !any(isnan, model(this_oneSecond_slice)) #CONV
                # push!(flattened_train_mfccs,flat_oneSecond_slice)
                push!(flattened_train_mfccs,this_oneSecond_slice) #CONV
                # push!(train_indices, counter)
                # println(model(flat_oneSecond_slice))
                # wag
            else
                global isnan_counter += 1
                # println(counter)
                push!(train_isnan_indices, oneSecondCounter)
                # println(train_isnan_indices)
            end
        end
        # wag
        # if !any(isnan, model(mfcc))
        #     push!(flattened_train_mfccs,mfcc)
        #     push!(train_indices, counter)
        # else
        #     global isnan_counter += 1
        #     # println(counter)
        #     push!(train_isnan_indices, counter)
        #     # println(train_isnan_indices)
        # end
    else
        global train_sizeCheckCounter += 1
        push!(train_incorrect_size_indices, counter)
    end
end
println(string(train_sizeCheckCounter, " additional mfcc's removed coz incorrect size"))
# println("Indices removed in Training set due to Nan values:")
# println(train_isnan_indices)

println("Current amount of NaN values detected:")
println(isnan_counter)
println("(of the one-second samples)")

println("Original size of training set (with NaN values):")
println(size(train_mfccs)[1])
println("New size (without NaN values & split in 4):")
println(size(flattened_train_mfccs)[1])

# vector_of_array = RecursiveArrayTools.VectorOfArray(flattened_train_mfccs) #CONV
# flattened_train_mfccs = convert(Array,vector_of_array) #CONV

# data = data'
# println(size(flattened_train_mfccs))
# wag
# println(train_mfccs)
# println(size(flattened_train_mfccs))
# println(train_mfccs[1])
# println(size(flattened_train_mfccs[1]))

training_data = flattened_train_mfccs
# println(size(training_data[1]))
# wag
println(size(training_data))
# wag
println(size(training_data[1][1]))
# wag
# sizeOfTraining = size(training_data)[5]
sizeOfTraining = size(training_data)[1]
println(string(sizeOfTraining), " = NR OF TRAINING SAMPLES")
# println(training_data[5,1])

# wag
#################################################################
# Get labels

train_labels = train_metaDF.class#[train_indices]
println(train_labels[10])
one_Second_train_labels = []
for label in train_labels
    for i in 1:4
        push!(one_Second_train_labels, label)
    end
end
# println(one_Second_train_labels)
train_labels = one_Second_train_labels
# println(train_labels[9*4+1:9*4+1+3])

filenames = train_metaDF.slice_file_name
# println(filenames[1:5])
one_second_filenames = []
for filename in filenames
    for i in 1:4
        push!(one_second_filenames, filename)
    end
end
println(one_second_filenames[1:20])
# wag
# println(train_isnan_indices)
# println(unique(train_isnan_indices))
# println(size(train_isnan_indices))
# println(train_labels[train_isnan_indices])
println(size(train_labels))
deleteat!(train_labels, train_isnan_indices)
println(size(train_labels))
println(size(train_incorrect_size_indices))
println(train_incorrect_size_indices)
fourSecond_train_incorrect_size_indices = []
for index in train_incorrect_size_indices
    # println(index)
    # index = 3
    scaled_start = (index-1)*4
    for i in 1:4
        # println(scaled_start+i)
        push!(fourSecond_train_incorrect_size_indices, scaled_start+i)
    end
    # wag
end
# wag
deleteat!(train_labels, fourSecond_train_incorrect_size_indices)
# deleteat!(train_labels, train_incorrect_size_indices)
println(size(train_labels))
# wag

# nrToDisplay = 10
# println(string(size(train_labels), " SIZE OF TRAINING LABELS"))
# println(train_metaDF.class[1:nrToDisplay])
# println(train_labels[1:nrToDisplay])
# println(string(size(test_labels), " SIZE OF TEST LABELS"))
# println(test_metaDF.class[1:nrToDisplay])
# println(test_labels[1:nrToDisplay])
# println("###################################################")

println(train_labels[1])
println(train_labels[100])
train_labels = onehotbatch(train_labels, classes)
println(train_labels[:,1])
println(train_labels[:,100])
# wag

# nrToDisplay = 5
println(string(size(train_labels), " SIZE OF TRAINING LABELS"))
# println(train_metaDF.class[1:nrToDisplay])
# println(train_labels[1:nrToDisplay*10])

# wag

print(size(test_mfccs))
println(" test mfccs considered (at first)")

flattened_test_mfccs = []
global counter = 0
global oneSecondCounter = 0
global test_sizeCheckCounter = 0
global test_isnan_indices = []
global test_incorrect_size_indices = []
for mfcc in test_mfccs
    global counter += 1
    if size(mfcc) == (398, 13)
        # mfcc = collect(Iterators.flatten(mfcc))
        mfcc = Flux.normalise(mfcc, dims=1) #normalise too
        my_zero_padding = zeros(2,13)
        # println(my_zero_padding)
        padded_mfcc = vcat(mfcc, my_zero_padding) #apply zero padding
        # splitting each mfcc in 4:
        for i in 1:4
            global oneSecondCounter += 1
            this_oneSecond_slice = padded_mfcc[(1+(i-1)*100:i*100),:]
            # println(size(this_oneSecond_slice))
            this_oneSecond_slice = reshape(this_oneSecond_slice, (size(this_oneSecond_slice)...,1))
            this_oneSecond_slice = reshape(this_oneSecond_slice, (size(this_oneSecond_slice)...,1))
            # wag
            # flat_oneSecond_slice = collect(Iterators.flatten(this_oneSecond_slice)) #CONV
            # println(size(flat_oneSecond_slice))
            # wag
            # if !any(isnan, model(flat_oneSecond_slice))
            if !any(isnan, model(this_oneSecond_slice))
                # push!(flattened_test_mfccs,flat_oneSecond_slice)
                push!(flattened_test_mfccs,[this_oneSecond_slice])
                # push!(test_indices, counter)
                # println(model(flat_oneSecond_slice))
                # wag
            else
                global isnan_counter += 1
                # println(counter)
                push!(test_isnan_indices, oneSecondCounter)
                # println(test_isnan_indices)
            end
        end
        # if !any(isnan, model(mfcc))
        #     push!(flattened_test_mfccs,mfcc)
        # else
        #     global isnan_counter += 1
        #     println(counter)
        #     push!(test_isnan_indices, counter)
        #     # println(test_isnan_indices)
        #     # wag
        # end
    else
        global test_sizeCheckCounter += 1
        push!(test_incorrect_size_indices, counter)
    end
end
println(test_isnan_indices)
# wag
println(string(test_sizeCheckCounter, " additional mfcc's removed coz incorrect size"))
println("Indices removed in Training set due to Nan values:")
println(test_isnan_indices)

print("Current nr of NaN values detected = ")
println(isnan_counter)

println("Old size of training set (with NaN values):")
println(size(test_mfccs))
println("New size (without NaN values):")
println(size(flattened_test_mfccs))

# vector_of_array = RecursiveArrayTools.VectorOfArray(flattened_test_mfccs)
# flattened_test_mfccs = convert(Array,vector_of_array)

test_data = flattened_test_mfccs
println(size(test_data))
println(size(test_data[1][1]))
# sizeOfTest = size(test_data)[5]
sizeOfTest = size(test_data)[1]
println(string(sizeOfTest), " = NR OF TEST SAMPLES")
# wag

test_labels = test_metaDF.class
one_Second_test_labels = []
for label in test_labels
    for i in 1:4
        push!(one_Second_test_labels, label)
    end
end
# println(one_Second_train_labels)
println(size(one_Second_test_labels))
println(size(test_labels))
test_labels = one_Second_test_labels
println(size(test_labels))
# wag
deleteat!(test_labels, test_isnan_indices)
fourSecond_test_incorrect_size_indices = []
for index in test_incorrect_size_indices
    # println(index)
    # index = 3
    scaled_start = (index-1)*4
    for i in 1:4
        # println(scaled_start+i)
        push!(fourSecond_test_incorrect_size_indices, scaled_start+i)
    end
    # wag
end
# wag
deleteat!(test_labels, fourSecond_test_incorrect_size_indices)
# deleteat!(test_labels, test_incorrect_size_indices)
println(size(test_labels))
test_labels = onehotbatch(test_labels, classes)
println(string(size(test_labels), " SIZE OF TEST LABELS"))
println(string(isnan_counter, " NaN values removed between training and test sets"))
# println(test_metaDF.class[1:nrToDisplay])
# println(test_labels[1:nrToDisplay*10])
# println(test_data[2]) # equivalent to test_data[1,2]

println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# wag

# println(params)
# println(training_data[1:2])
println(size(training_data))
println(size(training_data[10]))
println(size(training_data))
println(size(training_data[10]))
# println(labels[23])
# println(optimiser)

# data = [randn(5174), randn(5174), randn(5174)]
# labels = [randn(10), randn(10)]

# println(size(data))
# println(size(labels))

# correct_answer = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # the  correct_answer
# println(correct_answer)
# correct_answer = Flux.onehot("children_playing", classes)
# train_labels = fill(correct_answer, 50)
# labels = [correct_answer]

# println(train_labels)
println(size(train_labels))
println(size(train_labels[:,1]))
println(train_labels[:,1])
println(train_labels[:,2796])

# wag

# data = zip(training_data,train_labels)
# training_data = Float32.(training_data)
# test_data = Float32.(test_data)

# println(training_data)
# println(1:size(training_data)[2])
# wag
# println(training_data[1])
# println(size(training_data))
# println(size(training_data[:,:,:,:,1]))
# println(sizeOfTraining)
# fourD_rows = []
# # wag
# for k in 1:sizeOfTraining
#     # println(k)
#     fourD_row = training_data[:,:,:,:,k]
#     # println(size(fourD_row))
#     push!(fourD_rows,fourD_row)
#     # println(fourD_rows)
#     # println(size(fourD_rows))
#     # println(size(fourD_rows[1]))
#     # wag
# end
# println(size(fourD_rows))
# println(size(fourD_rows[3]))
# training_data = fourD_rows
# wag

println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
println(size(training_data))
println(typeof(training_data))
println(size(training_data[1]))
println(typeof(training_data[1]))
println(typeof(training_data[1][1]))
# println(typeof(training_data[2796][1]))
# wag

# training_data = convert(Array{Float32,4}, training_data)

array_of_tuples = []
for i in 1:sizeOfTraining
    # println(typeof(training_data[i]))
    training_data[i] = Float32.(training_data[i])
    # println(typeof(training_data[i]))
    # wag
    tuple = (training_data[i], train_labels[:,i])
    # tuple = (training_data[i], train_labels[:,i], one_second_filenames[i])
    # println(typeof(tuple))
    # println(tuple)
    # println(size(tuple[1]))
    push!(array_of_tuples, tuple)
    # println(array_of_tuples)
    # wag
end
training_set = array_of_tuples

# training_set = [(training_data, train_labels)]

# new_train_labels = []
# for i in 1:sizeOfTraining
#     push!(new_train_labels,train_labels[:,i])
#     # println(typeof(new_train_labels))
#     # println(new_train_labels)
#     # wag
# end
# training_set = zip(training_data,new_train_labels)

# println(size(training_set))
# println(size(training_set[1]))
# println(size(training_set))
# println(typeof(training_set))
# println(typeof(training_set[1]))
# # println(size(training_set[1]))
# println(size(training_set[1][1]))
# println(typeof(training_set[1][1]))
# println(size(training_set[1][2]))
# println(typeof(training_set[1][2]))
# wag
# println(size(training_set))
# println(size(training_set))
# wag
# println(size(training_set))

# println(training_data[1])
# println(size(training_data[1]))
# wag

println(size(train_labels))
println(length(train_labels))
println(size(train_labels[1]))
println(train_labels[1])
# println(train_labels[1,1])

# println(test_data[1])
# println(size(test_data))
# println(size(test_data[1]))
# # println(length(test_data[1]))
# wag

# test_set = [(test_data,test_labels)]
array_of_tuples = []
for i in 1:sizeOfTest
    # println(typeof(training_data[i]))
    # println(test_labels[:,i])
    # println(train_labels[:,i])
    # wag
    this_test_mfcc = Float32.(test_data[i][1])
    # println(typeof(training_data[i]))
    # wag
    tuple = (this_test_mfcc, test_labels[:,i])
    # tuple = (training_data[i], train_labels[:,i], one_second_filenames[i])
    # println(typeof(tuple))
    # println(tuple)
    # println(size(tuple[1]))
    push!(array_of_tuples, tuple)
    # println(array_of_tuples)
    # wag
end
test_set = array_of_tuples
# test_set = (test_data,test_labels)
# test_set = zip(test_data,test_labels)
# println(size(test_data))
# println(size(test_data[1]))
# println(size(test_data[1][1]))
# println(size(test_data[2][1]))
# println(size(test_data[468][1]))
# println(test_data[468][1][1:10])
# wag

i = rand(1:sizeOfTest)
# println(size(test_data)[5])
# println(i)
# println(size(test_data)[1])
println(string(i," random test sample's index"))
# println(size(test_data[i]))
test_input = test_data[i][1]
println(size(test_input))
# println(test_input)
test_input_label = test_labels[:,i]
test_input_label = onecold(test_input_label,classes)
println(test_input_label)
# wag

j = rand(1:sizeOfTest)
println(string(j," ANOTHER random test sample's index"))
another_test_input = test_data[j][1]
another_test_input_label = test_labels[:,j]
another_test_input_label = onecold(another_test_input_label,classes)

# training_input = training_mfccs.values[55]
k = rand(1:sizeOfTraining)
println(string(k," random training sample's index"))
println(size(training_data))
println(size(training_data[k]))
# wag
training_input = training_data[k]
training_input_label = train_labels[:,k]
training_input_label = onecold(training_input_label,classes)

println("hey")
println(size(test_input))
println(another_test_input[1:10])
println(training_input[1:10])
# wag

# model = load("model.jld", "SimpleModel2")
# @load "mySimpleModel2.bson" model
# println(model)
# println(sizeof(Flux.params(model)))
# @load "myWeights.bson" weights
# Flux.loadparams!(model,weights)

# wag
println("##########################################")
println("##########################################")
println(size(test_input))
println(findmax(test_input))
println(findmin(test_input))
println(typeof(test_input))
println(test_input[20])
# println(size(model(test_input)))

predictionBefore = model(test_input)
println(string(predictionBefore, " (PREDICTION BEFORE TRAINING)"))
println(string("(",onecold(predictionBefore, classes),")"))
display(Plots.bar(classes, predictionBefore, size=(400,300), dpi=40, title="BEFORE TRAINING", ylims=(0,1), xrotation=45, legend=:topright))
# wag

predictionBefore = model(another_test_input)
println(string(predictionBefore, " (ANOTHER PREDICTION BEFORE TRAINING)"))
println(string("(",onecold(predictionBefore, classes),")"))
display(Plots.bar(classes, predictionBefore, size=(400,300), dpi=40, title="ALSO BEFORE TRAINING", ylims=(0,1), xrotation=45, legend=:topright))

# function avg_loss(data, model)
#     total_loss = 0
#     sizeOfData = size(data)[1]
#     # println(sizeOfData)
#     # wag
#     for (x,y) in data
#         # println(size(x))
#         # println(size(x)[1])
#         # println(y)
#         # this_prediction = model(x)
#         # println(this_prediction)

#         # augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))
#         # thisLoss = Flux.logitcrossentropy(model(augment(x)), y)
#         # println(thisLoss)

#         thisLoss = Flux.mse(model(x),y)
#         # println(thisLoss)

#         total_loss += thisLoss
#         # println(total_loss)
#         # wag
#     end
#     avg_total_loss = total_loss/sizeOfData
#     # println(avg_total_loss)
#     # wag
#     return avg_total_loss
# end


# wag


# function accuracy(data, model)
#     acc = 0
#     correct_predictions = 0
#     for (x,y) in data
#         # println(size(x))
#         # println(size(y))
#         if onecold(model(x),classes)[1] == onecold(y,classes)
#             correct_predictions += 1
#             # wag
#         end
#         # acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
#         # println(onecold(cpu(model(x)))[1])
#         # println(onecold(cpu(y))[1])
#         # println(onecold(cpu(model(x)))[1] .== onecold(cpu(y))[1])
#         # println(sum(onecold(cpu(model(x))) .== onecold(cpu(y))))
#         # println(size(x)[2])
#         # println(length(data))
#         # wag
#         acc = correct_predictions / size(data)[1]
#         # println(acc)
#         # println(acc/length(data))
#         # wag
#     end
#     # acc/length(data) # retarded? Note that length(data) always = 1, ffs?
#     acc
# end


penalty() = sum(abs2, model.W) + sum(abs2, model.b)
# sqnorm(x) = sum(abs2, x)
# penalty(model) = sum(sqnorm, params(model))

# loss(x,y) = Flux.logitcrossentropy(model(x), y) + penalty(model)
# loss(x,y) = Flux.logitcrossentropy(model(x), y)

# loss(x,y) = Flux.mse(model(x), y) + penalty()
# loss(x,y) = Flux.mse(model(x), y) + sum(sqnorm(x), params(model))
loss(x,y) = Flux.mse(model(x), y)

ƞ = 5e-5
# ƞ = 1e-4
optimiser = Flux.Optimise.ADAM(ƞ)
ps = params(model)

global training_losses = []
global training_accs = []
global test_accs = []

function myTrain!(loss, ps, training_set, test_set, optimiser)
    @time begin
        # println(size(training_set))
        # [9*4+1:9*4+1+3]
        # augment(x) = x .+ gpu(0.1f0*randn(eltype(x), size(x)))
        # loss(x,y) = Flux.logitcrossentropy(model(augment(x)), y)
        # loss(x,y) = Flux.mse(model(x), y)
        # for (x,y, f) in training_set[9*4+1:9*4+1+3]
        total_loss_train = 0
        correct_training_predictions = 0
        for d in training_set#[9*4+1:9*4+1+3]
            if onecold(model(d[1]),classes)[1] == onecold(d[2],classes)
                correct_training_predictions += 1
                # wag
            end
            # training_set[:,1] = Float32.(training_set[:,1])
            # println(size(training_set))
            # println(size(training_set[1][1]))
            # println(typeof(training_set[1][1]))
            # println(training_set[1][1][1:10])
            # wag
            # x = Float32.(x)
            # println(size(test_input))
            # println(size(another_test_input))
            # println(size(x))
            # println(size(y))
            # wag
            # thisLoss = loss(x,y)
            # this_prediction = model(x)
            # println("this Prediction:")
            # println(this_prediction)
            # println(onecold(this_prediction,classes))
            # println("Real Answer:")
            # println(y)
            # println(onecold(y,classes))
            # println("Filename:")
            # println(f)
            # thisLoss = Flux.logitcrossentropy(this_prediction, y)
            # println("This loss:")
            # println(thisLoss)
            # thisLoss = loss(d...)
            # println(this_loss)
            
            # println("total_loss_train:")
            # println(total_loss_train)

            local training_loss
            # ps = params(model)
            # println(size(ps[11]))
            # println("Old parameters:")
            # println(ps[11][1:10])
            # println(ps[12])
            # wag
            gs = gradient(ps) do
                # total_loss_train
                training_loss = loss(d...)
                # println(training_loss)
                # println("here")
                total_loss_train += training_loss
                # println(total_loss_train)
                # wag
                return training_loss
            end

            # println(gs)
            # println("here")
            # println(typeof(gs))
            # println(gs)
            Flux.update!(optimiser,ps,gs)
            # println("Updated parameters:")
            # println(ps[11][1:10])
            # wag
        end
        # wag
        # println(string(sizeOfData, " values considered for training"))
        train_accuracy = correct_training_predictions/sizeOfTraining
        println("Training accuracy")
        println(train_accuracy)
        push!(training_accs, train_accuracy)

        correct_test_predictions = 0
        for (x,y) in test_set
            if onecold(model(x),classes)[1] == onecold(y,classes)
                correct_test_predictions += 1
            end
        end
        test_accuracy = correct_test_predictions/sizeOfTest
        println("Test accuracy")
        println(test_accuracy)
        push!(test_accs, test_accuracy)
        
        # @show accuracy((training_set), model)
        # @show accuracy((test_set), model)
        
        total_loss_train /= sizeOfTraining
        print("Average training loss = ")
        println(total_loss_train)
        push!(training_losses, total_loss_train)
        # println(training_losses)
        # wag
        # @show(avg_loss(training_set, model))
        # @show(avg_loss(training_set[9*4+1:9*4+1+3], model))
    end
end

@time begin
Flux.@epochs 30 myTrain!(loss, ps, training_set, test_set, optimiser)
end

display(Plots.plot(training_losses, xlabel="Epochs", title="Training losses", size=(400,300), dpi=40, legend=:topleft))

# display(Plots.plot(training_accs, xlabel="Epochs", title="Training Accuracy", size=(400,300), dpi=40, ylims=(0,1), legend=:topleft))

display(Plots.plot([training_accs,test_accs], xlabel="Epochs", title="ACCURACY", size=(400,300), dpi=40, ylims=(0,1), legend=:topleft, labels=["Training accuracy" "Test accuracy"]))

println("END")
wag













# evalcb = () -> @show(loss_all(training_set, model, sizeOfTraining))

global train_accuracies = [] #train_accuracy after each epoch
global test_accuracies = [] #test_accuracy after each epoch
function evalcb()
    @show(loss_all(training_set, model, sizeOfTraining))
    for (x,y) in training_set
        train_accuracy = sum(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x)[2]
        # println(train_accuracies)
        # println(train_accuracy)
        push!(train_accuracies, train_accuracy)
        # println(train_accuracies)
    end
    @show(loss_all(test_set, model, sizeOfTest))
    for (x,y) in test_set
        test_accuracy = sum(onecold(cpu(model(x))) .== onecold(cpu(y))) / size(x)[2]
        # println(test_accuracies)
        # println(test_accuracy)
        push!(test_accuracies, test_accuracy)
        # println(test_accuracies)
    end
end

# sqnorm(x) = sum(abs2,x)
# penalty() = sum(abs2, m.W) + sum(abs2, m.b)
loss(x,y) = Flux.logitcrossentropy(model(x), y)# + sum(sqnorm, Flux.params(model))
# loss(x,y) = Flux.Losses.mse(model(x), y)

ps = Flux.params(model)

# println(loss(test_input,test_input_label))

#learning rate
ƞ = 0.001
# ƞ = 0.0001 #-> test loss of 1.757 at 50 epochs
# ƞ = 0.00001 #-> TOO SLOW.. makes very little progress, even after 200 epochs
# optimiser = Flux.Optimise.Descent(ƞ)
optimiser = Flux.Optimise.ADAM(ƞ)

# println(model(test_input))


#Node values
# println()
# println("Nodes (all initialised as zero):")
# ps = params(model)
# println(size(ps[2]))
# println(ps[2])
# println(size(ps[4]))
# println(ps[4])
# print(size(ps[6]))
# println(" (output layer):")
# println(ps[6])

#Weights
# println()
# println("Weights:")

# println()
# println("Weights connecting 1st layer with 2nd layer's 1st node (some of them):")
# println(size(ps[1]))
# println(ps[1][1,:][5140:5174]) # all weights connecting 1st layer with 2nd layer's 1st node
# println("Weights between 1st and 2nd layer:")
# println(findmax(ps[1][:,:]))
# println(findmin(ps[1][:,:]))

# println()
# println("Weights between 2nd and 3rd layer:")
# println(findmax(ps[3][:,:]))
# println(findmin(ps[3][:,:]))

# println()
# println("Weights between 3rd and output layer:")
# println(findmax(ps[5][:,:]))
# println(findmin(ps[5][:,:]))

# println()
# println("All weight matrices:")
# println(size(ps[1]))
# println(size(ps[3]))
# println(size(ps[5]))
# wag

println("##############################################")
println(size(training_set))
println(typeof(training_set))
println()
# println(size(training_set[1]))
println(typeof(training_set[1]))
println()
println(size(training_set[2796][1]))
println(typeof(training_set[2796][1]))
println()
# println(size(training_set[1][1][2796]))
# println(typeof(training_set[1][1][2796]))
println()
println(size(training_set[2796][2]))
println(typeof(training_set[2796][2]))

# wag

Flux.@epochs 100 Flux.Optimise.train!(loss, ps, training_set, optimiser,
cb = evalcb
# cb = Flux.throttle(() -> println("training"), 2)
# cb = Flux.throttle(evalcb, 5)
# cb = function ()
#     # accuracy() > 0.9 && Flux.stop()
#     prediction = model(test_input)
#     println(prediction)
#     println(test_input_label)
#     wag
#     # accuracy = @show(1-loss(prediction, correct_answer[1]))
#     # accuracy = @show(1-loss(test_input, correct_answer))
#     # accuracy = @show(1-loss(prediction), data)
#     # push!(train_accuracy, accuracy)
#   end
)

# wag
# weights = Flux.params(model)
# @save "myWeights.bson" weights

prediction = model(test_input)
println("PREDICTION AFTER TRAINING:")
println(prediction)

another_prediction = model(another_test_input)
println("ANOTHER PREDICTION:")
println(another_prediction)

trained_prediction = model(training_input)
println("PREDICTION USING TRAINING SET:")
println(trained_prediction)

# global train_accuracy = []
# global test_accuracy = []
# for test_input in test_data
#     accuracy = 1 - loss(test_input, correct_answer)
#     push!(accuracy, test_accuracy)
# end
# println(test_accuracy)

print("First prediction: ")
print(Flux.onecold(prediction, classes))
println(string(" (correct answer = ", test_input_label, ")"))
print("Another prediction: ")
print(Flux.onecold(another_prediction, classes))
println(string(" (correct answer = ", another_test_input_label, ")"))
print("TRAINED PREDICTION: ")
print(Flux.onecold(trained_prediction, classes))
println(string(" (correct answer = ", training_input_label, ")"))

display(Plots.bar(classes, prediction, title="TEST 1 (AFTER TRAINING)", size=(400,300), dpi=40, ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.bar(classes, another_prediction, title="TEST 2 (AFTER TRAINING)", size=(400,300), dpi=40, ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.bar(classes, trained_prediction, title="TRAINED PREDICTION", size=(400,300), dpi=40, ylims=(0,1), xrotation=45, legend=:topright))
display(Plots.plot([train_accuracies,test_accuracies], xlabel="Epochs", title="ACCURACY", size=(400,300), dpi=40, ylims=(0,1), legend=:topleft, labels=["Training accuracy" "Test accuracy"]))


# println("End of Program")
