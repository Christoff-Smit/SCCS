#JULIA PACKAGES
using CSV: read
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
    
    metadataDF = read(path_to_metadata,datarow=2) #first line = header, therefore data starts on line 2
    
    # metadataDF = groupby(metadataDF, 8)
    
    # columnNames = names(metadataDF)

    classes = unique(metadataDF.class)
    
    describe_DF(metadataDF, classes)

    println("Only considering 'dog_bark' for now:")
    metadataDF = filter(row -> row[:class] == "dog_bark", metadataDF)
    describe_DF(metadataDF, ["dog_bark"])

    # selectedIndices = (Footsteps=1,Rain=8,Wind=9,Engine=11,Glass=12,Squeek=18,Tearing=19) #FSDnoisy18k
    # println(selectedIndices.Engine)
    # selectedIndices = [1,8,9,11,12,18,19] #FSDnoisy18k
    # println(selectedIndices)
    # println(typeof(selectedIndices))
    
    # df = extractRelevantData(df, selectedIndices, classes) # if you don't want to use all of the classes in the dataset
    
    # determineClassDistrib(df,false)
    return metadataDF, classes, path_to_wav_files, path_to_metadata
end
    
function splitTrainTest(metadataDF,test_fold)
    println(string("Fold ",test_fold," to be used as test data (with the rest as training data)"))
    trainingDF = filter(row -> row[:fold] != test_fold, metadataDF)
    println("Training metadataset shape:")
    println(size(trainingDF))
    testDF = filter(row -> row[:fold] == test_fold, metadataDF)
    println("Test metadataset shape:")
    println(size(testDF))
    return trainingDF, testDF
end

function generate_mfccs(trainingDF, testDF, path_to_wav_files)
    # mfcc(x::Vector, sr=16000.0; wintime=0.025, steptime=0.01, numcep=13, lifterexp=-22, sumpower=false, preemph=0.97, dither=false, minfreq=0.0, maxfreq=sr/2, nbands=20, bwidth=1.0, dcttype=3, fbtype=:htkmel, usecmp=false, modelorder=0)
    # println(size(mfcc[1])) # a matrix of numcep columns with for each speech frame a row of MFCC coefficients
    # println(size(mfcc[2])) # the power spectrum computed with DSP.spectrogram() from which the MFCCs are computed
    # println(mfcc[3]) # a dictionary containing information about the parameters used for extracting the features.

    mfccs = Dict()
    for row in eachrow(testDF)
        path_to_wav = string(path_to_wav_files,"fold",row.fold,"/",row.slice_file_name)
        signal, fs = WAV.wavread(path_to_wav)
        MFCC_output = MFCC.mfcc(signal[:,1], fs; numcep=13)
        mfcc = MFCC_output[1]
        spectrogram = MFCC_output[2]
        # push!(mfccs, row.slice_file_name => mfcc)
        push!(mfccs, path_to_wav => mfcc)
    end
    save(string(path_to_wav_files,"test_mfccs.jld"), "mfccs", mfccs)

    mfccs = Dict()
    for row in eachrow(trainingDF)
        path_to_wav = string(path_to_wav_files,"fold",row.fold,"/",row.slice_file_name)
        signal, fs = WAV.wavread(path_to_wav)
        MFCC_output = MFCC.mfcc(signal[:,1], fs; numcep=13)
        mfcc = MFCC_output[1]
        spectrogram = MFCC_output[2]
        # push!(mfccs, row.slice_file_name => mfcc)
        push!(mfccs, path_to_wav => mfcc)
    end
    save(string(path_to_wav_files,"training_mfccs.jld"), "mfccs", mfccs)
end










################################################################################################
# desired_classes = ["voices", "dogs barking", "gunshots", "alarms/sirens", "shattering glass", "wind", "footsteps", "car engine", "car horn", "rain", "cough", "finger snapping", "keys jangling", "laughter", "knocking", "tearing", "squeeks", "drilling"]
