
include("import.jl")
include("signal_processing.jl")

#Import the UrbanSound8K dataset:
trainingDF, testDF, path_to_wav_files, path_to_metadata = importUrbanSound8K()

wag

path_to_wav = string(path_to_wav_files,"/fold2/97193-3-0-4.wav")
describe_WAV(path_to_wav)