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


function process_wav(filepath)
    println("\nSound sample to be processed:")
    println(filepath)
    
    y, fs = WAV.wavread(path_to_wav)

    nrOfChannels = length(y[1,:])

    println("Signal Data:")
    myDescribe(y)

    print("Sampling frequency: ")
    println(fs)

    plot_wav(y,fs,nrOfChannels)

    signal = y[:,1] #from now on consider only one (the first) channel
    # println(signal[1:10])

    # p = plot_Periodogram(signal)
    # plot_PSD(p)

    plot_Spectrogram(signal,"PyPlot",fs)
    # plot_Spectrogram(signal,"DSP")
    
end

# path_to_wav = "C:/Users/Christoff/Downloads/sine.wav"
path_to_wav = "C:/Users/Christoff/Downloads/glass_harp.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/piano_e6.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/noise.wav"

process_wav(path_to_wav)

######################################################################
######################################################################


