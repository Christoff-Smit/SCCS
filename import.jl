#JULIA PACKAGES
using CSV: read
using DataFrames: names, groupby, unique, first, filter
using WAV
using Statistics
# using Gadfly
using Plots: plot, plot!

#USER PACKAGES:
include("functions.jl")

# This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes. The sampling rate, bit depth, and number of channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).


function importUrbanSound8K()
    path_to_wav_files = "D:/EERI 474 - Final year project/sound_libraries/UrbanSound8K/audio/"
    path_to_metadata = "D:/EERI 474 - Final year project/sound_libraries/UrbanSound8K/metadata/UrbanSound8K.csv"
    
    metadata_DF = read(path_to_metadata,datarow=2) #first line = header, therefore data starts on line 2
    
    # metadata_DF = groupby(metadata_DF, 8)
    
    columns = names(metadata_DF)

    classes = unique(metadata_DF.class)
    
    describeDF(metadata_DF, classes)
    # println(filter(row -> row[:class] == "dog_bark", metadata_DF))

    #let's just consider dog_bark for now (1 000 samples)
    metadata_DF = filter(row -> row[:class] == "dog_bark", metadata_DF)
    println(first(metadata_DF,5))
    
    # selectedIndices = (Footsteps=1,Rain=8,Wind=9,Engine=11,Glass=12,Squeek=18,Tearing=19) #FSDnoisy18k
    # println(selectedIndices.Engine)
    # selectedIndices = [1,8,9,11,12,18,19] #FSDnoisy18k
    # println(selectedIndices)
    # println(typeof(selectedIndices))
    
    # df = extractRelevantData(df, selectedIndices, classes) # if you don't want to use all of the classes in the dataset
    
    # determineClassDistrib(df,false)

    trainingDF = filter(row -> row[:fold] != 1, metadata_DF)
    println(size(trainingDF))
    testDF = filter(row -> row[:fold] == 1, metadata_DF)
    println(size(testDF))
    wag


    return trainingDF, testDF, path_to_wav_files, path_to_metadata
end


trainingDF, testDF, path_to_wav_files, path_to_metadata = importUrbanSound8K()

# println(trainingDF)
# println(testDF)









################################################################################################
# desired_classes = ["voices", "dogs barking", "gunshots", "alarms/sirens", "shattering glass", "wind", "footsteps", "car engine", "car horn", "rain", "cough", "finger snapping", "keys jangling", "laughter", "knocking", "tearing", "squeeks", "drilling"]

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K.....csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K/"

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.meta/test.csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.audio_test/"