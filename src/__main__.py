from collections import OrderedDict
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
import scoring
import word

# The input file
INPUT_FILE = 'input.wav'
# The training data directory
TRAIN_DATA = 'data/'
# The dictionary file
DICTIONARY_FILE = 'dictionary.txt'
# The weights file
WEIGHTS_FILE = 'weights.pth'

# The lower end of the freqency range
LOW_FREQUENCY = 80
# The heigher end of the freqency range
HIGH_FREQUENCY = 500

# The width of the used frequency range
FREQUENCY_RANGE = HIGH_FREQUENCY - LOW_FREQUENCY

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(FREQUENCY_RANGE, 10, 2, batch_first=True, bidirectional=True)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        output, hn = self.rnn(x)
        return output



def train(dataloader, model, dict):
    '''
    Trains the model with the given data.
    '''

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        words = word.split(y)
        # TODO Associate each words to a vector, creating a matrix
        y = y.to(device)

        X = X.to(device)
        pred = model(X)
        loss = model.loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, dict):
    model.eval()
    loss = 0.
    wer = 0.
    with torch.no_grad():
        for X, y in dataloader:
            words = word.split(y)
            # TODO Associate each words to a vector, creating a matrix
            y = y.to(device)

            X = X.to(device)
            pred = model(X)
            loss += model.loss_fn(pred, y)
            wer += word_error_rate(pred, y)

    size = len(dataloader)
    if size > 0:
        loss /= size
        wer /= size
    print(f"Test: Avg loss: {loss:f}; Avg WER: {wer:f}")



def create_dict(dataloader):
    dict = OrderedDict()

    for _, (_, y) in enumerate(dataloader):
        words = word.split(y)
        for w in words:
            dict[w] = None

    return dict

def load_dict(file):
    dict = OrderedDict()
    f = open(file, "r")

    for l in f.readlines():
        dict[l] = None

    return dict



def get_spectrogram(waveform):
    '''
    Returns the spectrogram of the given sound.
    '''

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
    '''
    Shows a spectrogram.
    '''

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

# Loading dataset and dictionary
dataloader = dataset.SpeechDataset(TRAIN_DATA)
dictionary = OrderedDict()
if not os.path.isfile(DICTIONARY_FILE):
    print('No dictionary found. Creating one...')
    dictionary = create_dict(dataloader)
else:
    print('Loading dictionary...')
    dictionary = load_dict(DICTIONARY_FILE)

# Selecting between training and prediction
if len(sys.argv) > 1 and sys.argv[1] == '--train':
    model = Net()
    # Loading weights
    if os.path.isfile(WEIGHTS_FILE):
        print('Loading weights...')
        model.load_state_dict(torch.load(WEIGHTS_FILE))
    else:
        print('No weights found. Training from zero')

    print('Training...')
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model)
        test(dataloader, model) # TODO Use bootstrapping on the dataset
    print("Done!")

    print('Saving weights...')
    torch.save(model.state_dict(), WEIGHTS_FILE)

    print('Saving dictionary...')
    f = open(DICTIONARY_FILE, 'w')
    for w, _ in dictionary:
        f.write(w)
        f.write("\n")
else:
    if os.path.isfile(WEIGHTS_FILE):
        model = Net()
        print('Loading weights...')
        model.load_state_dict(torch.load(WEIGHTS_FILE))

        if os.path.isfile(INPUT_FILE):
            print('Reading file...')
            print('File info:', torchaudio.info(INPUT_FILE))
            waveform, sample_rate = torchaudio.load(INPUT_FILE)

            # Showing the spectrogram of the input sound
            spec = get_spectrogram(waveform)
            show_spectrogram(spec)

            print('Running model...')
            net_in = numpy.moveaxis(numpy.array(spec), 1, -1)
            model.eval()
            with torch.no_grad():
                result = model(net_in)
            # TODO Interpret result using the dictionary

            print('Result:', result)
        else:
            print('Input file not found!')
    else:
        print('No weight file found!')
