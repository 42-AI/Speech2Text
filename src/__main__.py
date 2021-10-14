import os.path
import sys

from torch import nn
import librosa
import matplotlib.pyplot as plt
import numpy
import torch
import torchaudio
import torchaudio.transforms as transforms

import dataset
import word

# The input file
INPUT_FILE = 'input.wav'
# The training data directory
TRAIN_DATA = 'data/'
# The weights file
WEIGHTS_FILE = 'weights.pth'

# The lower end of the freqency range
LOW_FREQUENCY = 80
# The heigher end of the freqency range
HIGH_FREQUENCY = 500

# The width of the used frequency range
FREQUENCY_RANGE = HIGH_FREQUENCY - LOW_FREQUENCY

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(FREQUENCY_RANGE, 10, 2, bidirectional=True)

    def forward(self, x):
        self.rnn(x)



def train(dataloader, model):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        words = word.split(y)
        # TODO



def get_spectrogram(waveform):
    spectrogram_transform = transforms.Spectrogram(
        n_fft=HIGH_FREQUENCY * 2,
        win_length=None,
        hop_length=HIGH_FREQUENCY,
        center=True,
        pad_mode='reflect',
        power=2.0,
    )

    return spectrogram_transform(waveform)

def show_spectrogram(spec):
    # TODO Support multiple channels?
    figure, axis = plt.subplots(1, 1)
    axis.set_title('Spectrogram')
    axis.set_ylabel('Frequency')
    axis.set_xlabel('Frame')
    im = axis.imshow(librosa.power_to_db(spec[0]), origin='lower', aspect='auto')
    figure.colorbar(im, ax=axis)
    plt.show()



# ------------------------------
#    Main
# ------------------------------

if len(sys.argv) > 1 and sys.argv[1] == '--train':
    model = Net()
    if os.path.isfile(WEIGHTS_FILE):
        print('Loading weights...')
        model.load_state_dict(torch.load(WEIGHTS_FILE))
    else:
        print('No weights found. Training from zero')

    dataloader = dataset.SpeechDataset(TRAIN_DATA)
    print('Training...')
    train(dataloader, model)

    print('Saving weights...')
    torch.save(model.state_dict(), WEIGHTS_FILE)
else:
    if os.path.isfile(WEIGHTS_FILE):
        model = Net()
        print('Loading weights...')
        model.load_state_dict(torch.load(WEIGHTS_FILE))

        if os.path.isfile(WEIGHTS_FILE):
            print('Reading file...')
            print('File info:', torchaudio.info(INPUT_FILE))
            waveform, sample_rate = torchaudio.load(INPUT_FILE)

            # Showing the spectrogram of the input sound
            spec = get_spectrogram(waveform)
            show_spectrogram(spec)

            print('Running model...')
            # TODO Support multiple channels?
            result = model(spec[0])
            print('Result:', result)
        else:
            print('Input file not found!')
    else:
        print('No weight file found!')
