#JULIA PACKAGES
using CSV
using DataFrames: names, groupby, unique, first, filter
using WAV
using Statistics
# using Gadfly
using Plots: plot, plot!
using JLD: save, load
using MFCC

#USER PACKAGES:
include("misc.jl")

# This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes. The sampling rate, bit depth, and number of channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).


function importUrbanSound8K()
    path_to_wav_files = "D:/EERI 474 - Final year project/sound_libraries/UrbanSound8K/audio/"
    path_to_metadata = "D:/EERI 474 - Final year project/sound_libraries/UrbanSound8K/metadata/UrbanSound8K.csv"
    
    meta_DF = CSV.read(path_to_metadata,datarow=2, DataFrame) #first line = header, therefore data starts on line/row 2
    
    # meta_DF = groupby(meta_DF, 8)
    
    # columnNames = names(meta_DF)

    classes = unique(meta_DF.class)
    
    describe_DF(meta_DF, classes)

    # println("Only considering 'dog_bark' for now:")
    # meta_DF = filter(row -> row[:class] == "dog_bark", meta_DF)
    # describe_DF(meta_DF, ["dog_bark"])
    
    # determineClassDistrib(meta_DF,true) #wanna draw a pie chart? true or false
    
    return meta_DF, classes, path_to_wav_files, path_to_metadata
end

# function splitTrainTest(meta_DF,test_fold)
#     println(string("Fold ",test_fold," to be used as test data (with the rest as training data)"))
#     meta_trainingDF = filter(row -> row[:fold] != test_fold, meta_DF)
#     println("Training metadataset shape:")
#     println(size(meta_trainingDF))
#     meta_testDF = filter(row -> row[:fold] == test_fold, meta_DF)
#     println("Test metadataset shape:")
#     println(size(meta_testDF))
#     return meta_trainingDF, meta_testDF
# end

function splitTrainTest(meta_DF, MFCCs, test_fold)
    println(string("Fold ",test_fold," to be used as test data (with the rest as training data)"))
    meta_trainingDF = filter(row -> row[:fold] != test_fold, meta_DF)
    println("Training metadataset shape:")
    println(size(meta_trainingDF))
    meta_testDF = filter(row -> row[:fold] == test_fold, meta_DF)
    println("Test metadataset shape:")
    println(size(meta_testDF))
    return train_mfccs, test_mfccs
end

# function generate_mfccs(meta_trainingDF, meta_testDF, path_to_wav_files)
#     # mfcc(x::Vector, sr=16000.0; wintime=0.025, steptime=0.01, numcep=13, lifterexp=-22, sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=sr/2, nbands=20, bwidth=1.0, dcttype=3, fbtype=:htkmel, usecmp=false, modelorder=0)
#     # println(size(mfcc[1])) # a matrix of numcep columns with for each speech frame a row of MFCC coefficients
#     # println(size(mfcc[2])) # the power spectrum computed with DSP.spectrogram() from which the MFCCs are computed
#     # println(mfcc[3]) # a dictionary containing information about the parameters used for extracting the features.

#     test_mfccs = Dict()
#     for row in eachrow(meta_testDF)
#         # println(row)
#         # wag
#         path_to_wav = string(path_to_wav_files,"fold",row.fold,"/",row.slice_file_name)
#         signal, fs = WAV.wavread(path_to_wav)
#         MFCC_output = MFCC.mfcc(signal[:,1], fs; numcep=13)
#         mfcc = MFCC_output[1]
#         spectrogram = MFCC_output[2]
#         push!(test_mfccs, path_to_wav => mfcc)
#         # println(test_mfccs.(path_to_wav => mfcc))
#         # println(test_mfccs.keys)
#         # println(size(test_mfccs.keys))
#         # println(test_mfccs.values)
#         # println(size(test_mfccs.values))
#         # wag
#     end
#     save(string(path_to_wav_files,"test_mfccs.jld"), "mfccs", test_mfccs)

#     mfccs = Dict()
#     for row in eachrow(meta_trainingDF)
#         path_to_wav = string(path_to_wav_files,"fold",row.fold,"/",row.slice_file_name)
#         signal, fs = WAV.wavread(path_to_wav)
#         MFCC_output = MFCC.mfcc(signal[:,1], fs; numcep=13)
#         mfcc = MFCC_output[1]
#         spectrogram = MFCC_output[2]
#         push!(mfccs, path_to_wav => mfcc)
#     end
#     save(string(path_to_wav_files,"train_mfccs.jld"), "mfccs", mfccs)
# end




function generate_mfccs(meta_DF, path_to_wav_files)
    # mfcc(x::Vector, sr=16000.0; wintime=0.025, steptime=0.01, numcep=13, lifterexp=-22, sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=sr/2, nbands=20, bwidth=1.0, dcttype=3, fbtype=:htkmel, usecmp=false, modelorder=0)
    # println(size(mfcc[1])) # a matrix of numcep columns with for each speech frame a row of MFCC coefficients
    # println(size(mfcc[2])) # the power spectrum computed with DSP.spectrogram() from which the MFCCs are computed
    # println(mfcc[3]) # a dictionary containing information about the parameters used for extracting the features.

    mfccs = Dict{String, Array{Float64,2}}()
    # mfccs = Dict{String, Array{Float64,2}}("first" => [99 99; 99 99])
    # push!(mfccs, "testPath" => [1 2; 3 4])
    # println(mfccs)
    # println(mfccs.keys)
    # println(mfccs.values)
    # println(mfccs["testPath"])
    # println(mfccs["first"])
    # wag
    excluded_counter = 0
    index = 0
    unsupported_compression_code_indices = [608]
    # unsupported_compression_code_indices = [4804,6247,6248,6249,6250,6251,6252,6253,8339] #real
    for row in eachrow(meta_DF)[8732-1000:8732]
        index += 1
        println(index)
        # println(row)
        path_to_wav = string(path_to_wav_files,"fold",row.fold,"/",row.slice_file_name)
        # println(path_to_wav)
        # println(typeof(path_to_wav)) # it's a string
        if index in unsupported_compression_code_indices
            println(string("index = ", index, " THEREFORE SKIPPING"))
        else
            signal, fs = WAV.wavread(path_to_wav)
            MFCC_output = MFCC.mfcc(signal[:,1], fs; numcep=13)
            mfcc = MFCC_output[1]
            # println(typeof(mfcc))
            # wag
            # spectrogram = MFCC_output[2]
            # push!(mfccs, path_to_wav => mfcc) #! don't do this..
    
            # if size(mfcc) == (398, 13)
            if row."end"-row.start == 4
                push!(mfccs, row.slice_file_name => mfcc)
                # println(string(row.slice_file_name," added"))
            else
                excluded_counter += 1
                println(string(row.slice_file_name, " not the right size (", excluded_counter, " files excluded)"))
            end
            # println(mfccs.keys) #weird
            # println(size(mfccs.keys)) #weird
            # println(typeof(mfccs))
            # println(mfccs[path_to_wav]) #value that corresponds to path_to_wav
            # println(mfccs[row.slice_file_name]) #value that corresponds to path_to_wav
            # println(mfccs)
            # save(string(path_to_wav_files,"MFCCs.jld"), mfccs)
            # MFCCs = load(string(path_to_wav_files,"MFCCs.jld"))
            # println(typeof(MFCCs))
            # println(MFCCs)
            # println(MFCCs[path_to_wav])
            # println(MFCCs[row.slice_file_name])
            # println(size(MFCCs[row.slice_file_name]))
            # println(row.slice_file_name)
            # wag
        end
        
    end
    # println(mfccs.values) #doesn't work -> LoadError: type Dict has no field values
    # wag
    # save(string(path_to_wav_files,"MFCCs.jld"), mfccs)
    save(string(path_to_wav_files,"MFCCs_small.jld"), mfccs)
end


