import colored_noise_utils as noiser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from convolution_reverb_file import convolution_reverb

first_rir_file = 'h_left.mat'
second_rir_file = 'h_right.mat'

TRAINING_INPUT_PATH = 'Datasets/RIRNoise_Train_Input'
TRAINING_OUTPUT_PATH = 'Datasets/RIRNoise_Train_Output'
TESTING_INPUT_PATH = 'Datasets/RIRNoise_Test_Input'

CLEAN_TRAINING_DIR = Path('Datasets/clean_trainset_28spk_wav')
CLEAN_TESTING_DIR = Path("Datasets/clean_testset_wav")
clean_training_dir_wav_files = sorted(list(CLEAN_TRAINING_DIR.rglob('*.wav')))
clean_testing_dir_wav_files = sorted(list(CLEAN_TESTING_DIR.rglob('*.wav')))
print("Total training samples:", len(clean_training_dir_wav_files))

print("Generating Training data!!!!!")
if not os.path.exists(TRAINING_INPUT_PATH):
    os.makedirs(TRAINING_INPUT_PATH)
if not os.path.exists(TRAINING_OUTPUT_PATH):
    os.makedirs(TRAINING_OUTPUT_PATH)


for audio_file in tqdm(clean_training_dir_wav_files):
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    first_RIR_noised_audio = convolution_reverb(audio_file=audio_file, ir_file=first_rir_file)
    noiser.save_audio_file(np_array=first_RIR_noised_audio, file_path='{}/{}'.format(TRAINING_INPUT_PATH,audio_file.name))

    second_RIR_noised_audio = convolution_reverb(audio_file=audio_file, ir_file=second_rir_file)
    noiser.save_audio_file(np_array=second_RIR_noised_audio, file_path='{}/{}'.format(TRAINING_OUTPUT_PATH,audio_file.name))


print("Generating Testing data")
if not os.path.exists(TESTING_INPUT_PATH):
    os.makedirs(TESTING_INPUT_PATH)

for audio_file in tqdm(clean_testing_dir_wav_files):
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    first_RIR_noised_audio = convolution_reverb(audio_file=audio_file, ir_file=first_rir_file)
    noiser.save_audio_file(np_array=first_RIR_noised_audio,
                           file_path='{}/{}'.format(TESTING_INPUT_PATH, audio_file.name))
