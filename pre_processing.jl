using DSP
# using Flux
# using SampledSignals
using WAV
using Plots
using PyPlot: specgram, cla, cm, imshow, title, xlabel, gcf, xticks, gca, ylabel
# using TensorFlow #! not compatible with Windows
using PyCall
# PyCall.pygui(true) #true by default
PyCall.pygui(:tk) #choose between -> :wx, :gtk (or :tk?), or :qt
# using ScikitLearn
using MFCC

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

    timeArr = plot_WAV(y,fs,nrOfChannels, path_to_wav)

    signal = y[:,1] #considering only one (the first) channel

    p = plot_Periodogram(timeArr,signal)

    spectrum, freqs, timeRange = plot_Spectrogram(signal,"PyPlot",fs)
    # spectrum, freqs, timeRange = plot_Spectrogram(signal,"DSP",fs)
    # spectrum, freqs, timeRange = plot_Spectrogram(signal,"MFCC",fs)

    MFCC_output = MFCC.mfcc(signal[:,1], fs; numcep=13)
    spectrogram = MFCC_output[2]
    # println(MFCC_output[3])
    println("Spectogram matrix (MFCC.jl):")
    println(size(spectrogram))
    
    # imshow(spectrogram', aspect="auto")
    # title("Spectrogram (DSP.jl via MFCC.jl)")
    # xlabel("Time (ms)")
    # ylabel("Frequency (kHz)")
    # gca().invert_yaxis()
    # display(gcf())
end

# path_to_wav = "C:/Users/Christoff/Downloads/sine.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/glass_harp.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/piano_e6.wav"
# path_to_wav = "C:/Users/Christoff/Downloads/noise.wav"

# describe_WAV(path_to_wav)

######################################################################
######################################################################

# function get_spectrogram(path_to_wav)
#     y, fs = WAV.wavread(path_to_wav)
#     signal = y[:,1] #considering only one (the first) channel
#     spectrum, freqs, timeRange, im = plot_Spectrogram(signal,"PyPlot",fs)
#     println("Spectrogram image matrix:")
#     println(string(size(spectrum)[1]," x ",size(spectrum)[2]," = ", length(spectrum)))
#     return spectrum, freqs
# end