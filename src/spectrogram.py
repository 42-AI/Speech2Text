import librosa
import matplotlib.pyplot as plt
import numpy
import torch
import torchaudio.transforms as transforms

# The lower end of the freqency range
LOW_FREQUENCY = 80
# The heigher end of the freqency range
HIGH_FREQUENCY = 500

# The width of the used frequency range
FREQUENCY_RANGE = HIGH_FREQUENCY - LOW_FREQUENCY

def get_spectrogram(waveform):
    '''
    Returns the spectrogram of the given sound.
    '''

    spectrogram_transform = transforms.Spectrogram(n_fft=HIGH_FREQUENCY * 2)

    return spectrogram_transform(waveform)

def show_spectrogram(spec):
    '''
    Shows a spectrogram.
    '''

    figure, axis = plt.subplots(1, 1)
    axis.set_title('Spectrogram')
    axis.set_ylabel('Frequency')
    axis.set_xlabel('Frame')
    im = axis.imshow(librosa.power_to_db(spec[0]), origin='lower', aspect='auto')
    figure.colorbar(im, ax=axis)
    plt.show()

def prepare(audio):
    '''
    Turns the given Torch audio into a matrix ready to be fed into the RNN.
    '''

    waveform, sample_rate = audio
    spec = get_spectrogram(waveform)[0]
    X = numpy.expand_dims(numpy.array(spec), axis=0)
    X = numpy.moveaxis(X, 1, -1)
    # TODO X = X.view(X.shape[0], X.shape[1], LOW_FREQUENCY + 1)
    return torch.tensor(X)
