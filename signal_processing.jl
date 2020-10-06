using DSP
# using Flux
# using SampledSignals
using WAV
using Plots
using PyPlot: specgram
# using TensorFlow #! not compatible with Windows
using PyCall
# PyCall.pygui(true) #true by default
PyCall.pygui(:tk) #choose between -> :wx, :gtk (or :tk?), or :qt

include("functions.jl")


######################################################################
######################################################################


function process_wav(filepath)
    println("\nSound sample to be processed:")
    println(filepath)
    
    y, fs = WAV.wavread(path_to_wav)

    println("Signal Data:")
    myDescribe(y)
    print("Sampling frequency: ")
    println(fs)

    plot_wav(y,fs)

    signal = y[:,1] #consider the first channel
    # println(signal[1:10])

    # p = plot_Periodogram(signal)
    # plot_PSD(p)

    plot_Spectrogram(signal,"PyPlot")
    
end

# path_to_wav = "C:/Users/Christoff/Downloads/sine.wav"
path_to_wav = "C:/Users/Christoff/Downloads/glass_harp.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/piano_e6.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/noise.wav"

process_wav(path_to_wav)

######################################################################
######################################################################


