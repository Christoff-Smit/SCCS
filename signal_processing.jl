using DSP
# using Flux
# using SampledSignals
using WAV
using Plots
using PyPlot: specgram, cla
# using TensorFlow #! not compatible with Windows
using PyCall
# PyCall.pygui(true) #true by default
PyCall.pygui(:tk) #choose between -> :wx, :gtk (or :tk?), or :qt
# using ScikitLearn

include("functions.jl")
cla()

######################################################################
######################################################################


function describe_WAV(path_to_wav)
    # println("\nSound sample to be described:")
    # println(filepath)
    
    y, fs = WAV.wavread(path_to_wav)

    nrOfChannels = length(y[1,:])

    println("###########################")
    println("Signal Info:")
    println("###########################")
    myDescribe(y)
    print("Nr of channels: ")
    println(nrOfChannels)
    print("Sampling frequency: ")
    println(fs)
    println("###########################")

    t = plot_wav(y,fs,nrOfChannels)

    signal = y[:,1] #from now on consider only one (the first) channel
    # println(signal[1:10])

    p = plot_Periodogram(t,signal)

    spectrum, freqs, timeRange = plot_Spectrogram(signal,"PyPlot",fs)
    # plot_Spectrogram(signal,"DSP",fs)
end

path_to_wav = "C:/Users/Christoff/Downloads/sine.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/glass_harp.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/piano_e6.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/noise.wav"

# describe_WAV(path_to_wav)

######################################################################
######################################################################


