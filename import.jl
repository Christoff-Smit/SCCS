#JULIA PACKAGES
using CSV: read
using DataFrames: names, groupby, unique, first, filter
using WAV
using Statistics
# using Gadfly
using Plots: plot, plot!

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

    println("let's just consider 'dog_bark' for now:")
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
    println(string("Fold nr. ",test_fold," to be used as test data (with the rest as training data)"))
    trainingDF = filter(row -> row[:fold] != test_fold, metadataDF)
    println("Training dataset shape:")
    println(size(trainingDF))
    testDF = filter(row -> row[:fold] == test_fold, metadataDF)
    println("Test dataset shape:")
    println(size(testDF))
    return trainingDF, testDF
end




# println(trainingDF)
# println(testDF)









################################################################################################
# desired_classes = ["voices", "dogs barking", "gunshots", "alarms/sirens", "shattering glass", "wind", "footsteps", "car engine", "car horn", "rain", "cough", "finger snapping", "keys jangling", "laughter", "knocking", "tearing", "squeeks", "drilling"]

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K.....csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K/"

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.meta/test.csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.audio_test/"