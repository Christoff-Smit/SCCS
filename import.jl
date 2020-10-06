#JULIA PACKAGES
using CSV: read
using DataFrames
using WAV
using Statistics
# using Gadfly
using Plots: plot, plot!

#USER PACKAGES:
include("functions.jl")


function importFSDnoisy18k()
    path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDnoisy18k.meta/test.csv"
    path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDnoisy18k.audio_test/"
    
    df = read(path_to_metadata,datarow=2)
    
    classes = DataFrames.unique(df.label)
    
    describeDF(df, classes)
    
    # selectedIndices = (Footsteps=1,Rain=8,Wind=9,Engine=11,Glass=12,Squeek=18,Tearing=19) #FSDnoisy18k
    # println(selectedIndices.Engine)
    selectedIndices = [1,8,9,11,12,18,19] #FSDnoisy18k
    println(selectedIndices)
    println(typeof(selectedIndices))
    
    df = extractRelevantData(df, selectedIndices, classes)
    
    # determineClassDistrib(df,false)

    testDF = df
    trainingDF = df

    return trainingDF, testDF, path_to_wav_files, path_to_metadata
end


trainingDF, testDF, path_to_wav_files, path_to_metadata = importFSDnoisy18k()

# println(trainingDF)
# println(testDF)









################################################################################################
# desired_classes = ["voices", "dogs barking", "gunshots", "alarms/sirens", "shattering glass", "wind", "footsteps", "car engine", "car horn", "rain", "cough", "finger snapping", "keys jangling", "laughter", "knocking", "tearing", "squeeks", "drilling"]

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K.....csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K/"

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.meta/test.csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.audio_test/"