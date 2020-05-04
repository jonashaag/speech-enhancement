import numpy as np
import sys
import os
import soundfile as sf
import scipy.signal
import random
import pyloudnorm
import librosa

voices = open(sys.argv[1]).read().strip().splitlines()
#irs = open(sys.argv[2]).read().strip().splitlines()
#irs = [l.split(';')[0] for l in open(sys.argv[2]).read().strip().splitlines() if float(l.split(';')[-1])>=0.5]
irs = [l.split(';')[0] for l in open(sys.argv[2]).read().strip().splitlines() if l.endswith('y')]
out = open(sys.argv[3], "a")

random.shuffle(irs)

#import tensorflow as tf
#model = tf.keras.models.load_model('small-rooms.model')

while 1:
    voice = random.choice(voices)
    voicew = sf.read(voice)[0]
    ir = irs.pop()
    print(ir)
    irw = librosa.core.load(ir,sr=16000,mono=False)[0]
    if len(irw.shape) > 1:
        chan = random.randint(0, irw.shape[0]-1)
        irw = irw[chan]
    else:
        chan = -1
    irw = irw[:32000]
    ird = scipy.signal.convolve(voicew, irw)[len(irw):len(voicew)]
    ird = pyloudnorm.normalize.peak(ird, -10)
    if 0:
        pred = model.predict(np.expand_dims(
            np.abs(librosa.core.stft(np.asfortranarray(librosa.util.fix_length(irw,32000)),1024))
            ,0
        ))[0]
    else:
        pred = -1
    sf.write("/tmp/wav.wav", ird, samplerate=16000)
    os.system("ffplay -nodisp /tmp/wav.wav 2>/dev/null")
    choice = input(f"Room? {pred}")
    out.write(f"{ir};{chan};{choice}\n")
    out.flush()
