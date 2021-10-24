from collections import OrderedDict
import os.path
import sys

from torch import nn
import numpy
import torch
import torchaudio

import dataset
import scoring
import spectrogram
import word

# The input file
INPUT_FILE = 'input.wav'
# The training data directory
TRAIN_DATA = 'data/'
# The dictionary file
DICTIONARY_FILE = 'dictionary.txt'
# The weights file
WEIGHTS_FILE = 'weights.pth'

# FIXME device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("Using {} device".format(device))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(spectrogram.FREQUENCY_RANGE, 1, 2, batch_first=True, bidirectional=True)
        self.hn = torch.randn(4, 1, 1)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        self.hn = self.hn.to(device)
        y, hn = self.rnn(x, self.hn)
        self.hn = hn

        # TODO

        return y



def train(dataloader, model, dict):
    '''
    Trains the model with the given data.
    '''

    model.train()
    for batch in range(len(dataloader)):
        (X, y) = dataloader[batch]

        X = spectrogram.prepare(X)

        words = word.split(y)
        y = word.build_matrix(dict, words)

        X, y = X.to(device), y.to(device)
        pred = model(X)
        print(pred.shape, y.shape)
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
            X = spectrogram.prepare(X)

            words = word.split(y)
            y = word.build_matrix(dict, words)

            X, y = X.to(device), y.to(device)
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

    i = 0
    for batch in range(len(dataloader)):
        (_, y) = dataloader[batch]

        words = word.split(y)
        for w in words:
            dict[w] = i # TODO Do not insert if the dictionnary already contains the word
            i += 1

    return dict

def load_dict(file):
    dict = OrderedDict()
    f = open(file, "r")

    i = 0
    for l in f.readlines():
        dict[l] = i
        i += 1

    return dict



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
        train(dataloader, model, dictionary)
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
            spec = spectrogram.get_spectrogram(waveform)
            spectrogram.show_spectrogram(spec)

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
