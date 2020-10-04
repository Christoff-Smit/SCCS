#JULIA PACKAGES
using CSV: read
using DataFrames
using DSP: spectrogram, time, freq, power
using WAV
using Statistics
# using Gadfly
using Plots: plot
# using PyPlot
# using ScikitLearn
# using StatsModels
# using MultivariateStats
using Flux
# using Keras
using SampledSignals

#USER PACKAGES:
include("import-functions.jl")


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

file = trainingDF[rand(1:length(trainingDF[1])), :].fname
y, fs = process_with_SampledSignals(trainingDF,testDF,file)
display(plot(0:1/fs:(length(y)-1)/fs, y))
# xlabel("Time [s]")
println(fs)
println(y[2][1])
signal = y[:,1]
n = div(length(signal),8)
noverlap = div(n,2)
# println(y[:,1])
# println(y[:,sf])
SG = spectrogram(signal,n,noverlap)

display(SG)
println(SG.freq[1])
println(SG.power[1])
println(SG.time)
println(dfeqrer)
t = time(SG)
f = freq(SG)
imshow(flipud(log10(power(SG))), extent=[first(t), last(t), fs*first(f), fs*last(f)], aspect="auto")

println(dqerqwe)









################################################################################################
# desired_classes = ["voices", "dogs barking", "gunshots", "alarms/sirens", "shattering glass", "wind", "footsteps", "car engine", "car horn", "rain", "cough", "finger snapping", "keys jangling", "laughter", "knocking", "tearing", "squeeks", "drilling"]

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K.....csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/UrbanSound8K/"

# path_to_metadata = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.meta/test.csv"
# path_to_wav_files = "C:/Users/Christoff/Desktop/Uni/EERI 474 - Final year project/sound_libraries/FSDKaggle2018.audio_test/"