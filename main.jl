using JLD: save, load

include("import.jl")
include("pre_processing.jl")

#Import the UrbanSound8K dataset:
metadataDF, classes, path_to_wav_files, path_to_metadata = importUrbanSound8K()

test_fold = 1
trainingDF, testDF = splitTrainTest(metadataDF,test_fold)

#Perform Pre-Processing on the data:
# path_to_wav = string(path_to_wav_files,"/fold2/97193-3-0-0.wav")
# path_to_wav = string(path_to_wav_files,"/fold2/97193-3-0-1.wav")
# path_to_wav = string(path_to_wav_files,"/fold2/97193-3-0-4.wav")
# path_to_wav = string(path_to_wav_files,"/fold2/97193-3-0-6.wav")

# describe_WAV(path_to_wav)

spectrograms = []
for row in eachrow(trainingDF)
    path_to_wav = string(path_to_wav_files,"/fold",row.fold,"/",row.slice_file_name)
    spectrogram_matrix = get_spectrogram(path_to_wav)
    push!(spectrograms, spectrogram_matrix)
end
# println(spectrograms[2][123])
# println(size(spectrograms))
# println(sizeof(spectrograms))
# println(typeof(spectrograms))

save(string(path_to_wav_files,"dog_bark_training_spectros.jld"), "spectrograms", spectrograms)


trainingSpectros = load(string(path_to_wav_files,"dog_bark_training_spectros.jld"), "spectrograms")
println("====================================================================")
println(trainingSpectros[2][123])
println(size(trainingSpectros))
println(sizeof(trainingSpectros))
println(typeof(trainingSpectros))