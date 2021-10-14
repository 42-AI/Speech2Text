import os

import torch
import torchaudio

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str):
        '''
        `dir_path` is the path to the data directory.
        '''

        self.dir_path = dir_path
        self.speech_files = os.listdir(dir_path + '/speech/')

    def __getitem__(self, idx):
        speech_file = self.speech_files[idx]
        speech = torchaudio.load(speech_file)

        text_file = open(dir_path + '/text/' + str(idx), "r")
        text = text_file.read()

        return speech, text
