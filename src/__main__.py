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



def train(dataloader, model):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        words = word.split(y)

        X = X.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



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
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model)
        # TODO test(test_dataloader, model)
    print("Done!")

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
            #show_spectrogram(spec)

            print('Running model...')
            result = model(numpy.moveaxis(numpy.array(spec), 1, -1))
            print('Result:', result)
        else:
            print('Input file not found!')
    else:
        print('No weight file found!')
