
using PyPlot: specgram, show, plot, subplots, gcf, title, cm, xlabel, ylabel, xticks

################################################################################################
#FUNCTIONS:

################################################################################################
#for import.jl

function describeDF(df, classes)
    println(string(nrow(df)," entries for ",string(length(classes), " classes:")))
    println(classes)

    # println(df)
    # println(typeof(df))
    # println(DataFrames.describe(df))
    # println(DataFrames.groupby(df, "label"))
end

function extractRelevantData(df, selectedIndices, classes)
    df = DataFrames.groupby(df, :label)[selectedIndices] #extracted data
    # println(typeof(df))
    df = DataFrames.combine(df,:, keepkeys=false, ungroup=true) #convert grouped data into a dataframe
    # println(typeof(df))
    # println(df)
    println(string(nrow(df)," entries extracted from full dataset"))
    # println(string(nrow(df))*" entries extracted from full dataset")
    println("Extracted classes:")
    println(classes[selectedIndices])
    return df
end

function getSampleLengths(df)
    sample_lengths = Float32[]
    
    # Plot some sound waves:
    # for row in eachrow(last(df,5))
    #     path_to_wav = string(path_to_wav_files, row.fname)
    #     y, fs = WAV.wavread(path_to_wav) #signal data y, and sampling rate fs
    #     display(Plots.plot(0:1/fs:(length(y)-1)/fs,y,title=row.label))
    # end
    # println(noerer)
    
    for row in eachrow(df)
        path_to_wav = string(path_to_wav_files,row.fname)
        # println(path_to_wav)
        y, fs = WAV.wavread(path_to_wav) #signal data y, and sampling rate/frequency fs
        # WAV.wavplay(path_to_wav) #only works on Linux
        sample_length_in_secs = length(y)/fs
        push!(sample_lengths,sample_length_in_secs)
        # println(string(sample_length_in_secs, " seconds"))
        # println(length(y))
        # println(typeof(y))
        # println(size(y))
        # println(y[1])
        # println(y[length(y)])
        # println(fs)
    end
    
    df."Length" = sample_lengths
    # println(df)    
end

function determineClassDistrib(df,pie_chart)
    getSampleLengths(df)

    df_per_class = DataFrames.groupby(df,:label)

    avgs = []
    for class in df_per_class
        thisMean = Statistics.mean(class.Length)
        push!(avgs,thisMean)
    end
    
    println(string("Average sample length for each class:\n",avgs))
    
    if pie_chart == true
        # ann = ([1,-1],[1,-1],classes[selectedIndices])
        display(Plots.pie(classes[selectedIndices],avgs,title="Class Distribution\n(i.t.o. sample length)",
        # annotations=ann,
        l=0.5))
    end
end


function process_with_SampledSignals(trainingDF,testDF,file)
    # println(path_to_wav_files)
    # println(file)
    path_to_wav = string(path_to_wav_files,file)
    println("\nSound sample to be processed with SampledSignals:")
    print(file)
    println(string(" (",path_to_wav,")"))

    y, fs = WAV.wavread(path_to_wav) #signal data y, and sampling rate/frequency fs

    buf = SampledSignals.SampleBuf(y,fs) # 'data' and 'samplerate'
    # println(typeof(buf))
    # println(buf.samplerate)
    # println(buf[10])
    # println(y[10])

    source = SampledSignals.SampleBufSource(buf)
    sink = SampledSignals.SampleBufSink(buf)
    display(SampledSignals.read(source))
    # println(buf[100:200])
    # println(typeof(source))
    # OR
    # SampledSignals.write(sink,source)
    # println(typeof(source))


    # println(buf)
    # println(source)

    # spec = SampledSignals.fft(buf)
    # plot(SampledSignals.domain(spec), abs(spec))
    return y, fs
end

function myDescribe(object)
    println("Here follows the dimensions for:")
    display(object)
    print("typeof = ")
    println(typeof(object))
    print("sizeof = ")
    println(sizeof(object))
    print("size = ")
    println(size(object))
    print("length = ")
    println(length(object))
end
    
################################################################################################
# for signal-processing.jl

function plot_wav(y,fs,nrOfChannels)
    channels = 1:nrOfChannels

    start_time = 0
    period = 1/fs
    stop_time = (length(y)-1)/fs

    timeArr = start_time:period:stop_time/nrOfChannels

    if nrOfChannels>1
        Plots.plot(
            timeArr,y[:,1],
            label="channel 1",
            xlabel="Time (period intervals)",
            ylabel="Amplitude",
            title="Amplitude over time"
        )
        for channel in channels[2,:] #for every channel except the first one
            signal = y[:,channel]
            # signal = y
            display(Plots.plot!(
                timeArr,signal,
                label=string("channel ",channel)
                )
            )
        end
    else
        display(
            Plots.plot(
            timeArr,y[:,1],
            label="channel 1",
            xlabel="Time (s)",
            ylabel="Amplitude",
            title="Amplitude over time"
        ))
    end
    return timeArr
end

function plot_Periodogram(t,signal)
    periodogram = DSP.Periodograms.periodogram(signal)
    # println(periodogram) #! BAD IDEA
    # println(length(periodogram.power))
    # println(length(periodogram.freq))
    # println(periodogram[:,3])
    display(Plots.plot(
        periodogram.power,
        # xscale=:log10,
        xlims=(0,20000),
        xlabel="Frequency",
        ylabel="Amplitude (dB)",
        yscale=:log10,
        title="Power Density Spectrum"))
    # display(Plots.plot(periodogram.freq,title="Frequency"))
    # println(dferqwerwesdaf)
    return periodogram.power
end

function plot_Spectrogram(signal,framework,fs)
    if framework == "PyPlot"
        spectrum, freqs, timeRange = specgram(signal,Fs=fs,cmap=cm.inferno,noverlap=128,NFFT=512)
        # println(length(signal))
        # p -> powerSpectrum = 2-D array: Columns are the periodograms of successive segments.
        # freqs = 1-D array: The frequencies corresponding to the rows in spectrum.
        title("Spectrogram")
        xlabel("Time (s)")
        # show()
        display(gcf())
        return spectrum, freqs, timeRange
    elseif framework == "DSP"
        #default values:
        # n = div(length(signal),8)
        # noverlap = div(n,10)
        n = 512
        noverlap = 128
        
        # SG = DSP.Periodograms.spectrogram(signal,n,noverlap,fs=fs)
        SG = DSP.Periodograms.spectrogram(signal,fs=fs)
        # myDescribe(SG)
        
        t = DSP.time(SG)
        println("Spectrogram TIME VALUES")
        myDescribe(t)
        
        f = DSP.freq(SG)
        println("Spectrogram FREQUENCY VALUES")
        myDescribe(f)
        
        p = DSP.power(SG)
        println("Spectrogram POWER VALUES")
        myDescribe(p)
        # println(p[1])
        # println(p[2])
        # println(p[1,:])
        # println(t)
        # println(length(t))
        # println(dferqwerasdf)
        display(
            Plots.plot(p,
            xlabel="Time",
            # xscale=:log10,
            ylabel="Frequency",
            yscale=:log10,
            title="Spectrogram")
        )
    end
end

################################################################################################

