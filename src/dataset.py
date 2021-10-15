import os

import torch
import torchaudio

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str):
        '''
        `dir_path` is the path to the data directory.
        '''

        self.dir_path = dir_path

    def __len__(self):
        return len(os.listdir(self.dir_path + '/speech/'))

    def __getitem__(self, idx):
        speech_file = self.dir_path + '/speech/' + str(idx) + '.wav'
        speech = torchaudio.load(speech_file)

        text_file = open(self.dir_path + '/text/' + str(idx), "r")
        text = text_file.read()

        return speech, text
