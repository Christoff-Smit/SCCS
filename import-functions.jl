

################################################################################################
#FUNCTIONS:
################################################################################################

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