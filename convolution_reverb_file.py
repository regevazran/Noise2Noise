

import numpy as np
from scipy import signal
import sys
import os
import soundfile as sf

# Load support lib
from supplib import ReadList
from supplib import copy_folder
from supplib import load_IR
from supplib import shift

def convolution_reverb(audio_file, ir_file):
    # Open clean wav file
    [signal_clean, fs] = sf.read(audio_file)

    signal_clean = signal_clean.astype(np.float64)

    # Signal normalization
    signal_clean = signal_clean / np.max(np.abs(signal_clean))

    # Open Impulse Response (IR)
    IR = load_IR(ir_file)

    # IR normalization
    IR = IR / np.abs(np.max(IR))
    p_max = np.argmax(np.abs(IR))
    signal_rev = signal.fftconvolve(signal_clean, IR, mode="full")
    p_max = 120
    # Normalization
    signal_rev = signal_rev / np.max(np.abs(signal_rev))

    # IR delay compensation
    signal_rev = shift(signal_rev, -p_max)

    # Cut reverberated signal (same length as clean sig)
    signal_rev = signal_rev[0 : signal_clean.shape[0]]
    return signal_rev

