import os.path
import sys

from torch import nn
import torch
import torchaudio

# The input file
INPUT_FILE = 'input.mp4'
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

def main():
	if len(sys.argv) > 1 and sys.argv[1] == '--train':
		model = Net()
		if os.path.isfile(WEIGHTS_FILE):
			model.load_state_dict(torch.load(WEIGHTS_FILE))

		# TODO Train
	else:
		if os.path.isfile(WEIGHTS_FILE):
			model = Net()
			model.load_state_dict(torch.load(WEIGHTS_FILE))

			# TODO Run
		else:
			print('No weight file found!')

main()
