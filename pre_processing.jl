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

include("misc.jl")
cla()

######################################################################
######################################################################


function describe_WAV(path_to_wav)
    # println("\nSound sample to be described:")
    # println(filepath)
    
    y, fs = WAV.wavread(path_to_wav)

    nrOfChannels = length(y[1,:])

    println("#################################################")
    println("Signal Info:")
    myDescribe(y, path_to_wav)
    print("Nr of channels: ")
    println(nrOfChannels)
    print("Sampling frequency: ")
    println(fs)
    println("#################################################")

    timeArr = plot_WAV(y,fs,nrOfChannels)

    signal = y[:,1] #considering only one (the first) channel

    p = plot_Periodogram(timeArr,signal)

    spectrum, freqs, timeRange = plot_Spectrogram(signal,"PyPlot",fs)
    # plot_Spectrogram(signal,"DSP",fs)

    # println(spectrum[1:5, :])
    # println(typeof(spectrum))
    # println(sizeof(spectrum))
    println("Spectogram image matrix:")
    println(string(size(spectrum)[1]," x ",size(spectrum)[2]," = ", length(spectrum)))

    # println(freqs[1:1])
    # println(typeof(freqs))
    # println(sizeof(freqs))
    # println(size(freqs))
end

# path_to_wav = "C:/Users/Christoff/Downloads/sine.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/glass_harp.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/piano_e6.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/noise.wav"

# describe_WAV(path_to_wav)

######################################################################
######################################################################

function get_spectrogram(path_to_wav)
    y, fs = WAV.wavread(path_to_wav)
    signal = y[:,1] #considering only one (the first) channel
    spectrum, freqs, timeRange, im = plot_Spectrogram(signal,"PyPlot",fs)
    Plots.im
    println("Spectrogram image matrix:")
    println(string(size(spectrum)[1]," x ",size(spectrum)[2]," = ", length(spectrum)))
    return spectrum
end