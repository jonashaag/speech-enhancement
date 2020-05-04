import sys
import numpy as np
import tensorflow as tf
import librosa
import tqdm
import tqdm.contrib.concurrent
import random

model = tf.keras.models.load_model('small-rooms-magspec.model')

out = open(sys.argv[2], 'w')

def eval_(f):
    r, _ = librosa.core.load(f,sr=16000,mono=False)
    if len(r.shape) < 2:
        r = [r]
    s = np.array([
        np.abs(librosa.core.stft(np.asfortranarray(librosa.util.fix_length(w,32000)),1024))
        for w in r
    ])
    for i, p in enumerate(model.predict(s)):
        out.write(f"{f};{i};{p[0]}\n")

files = open(sys.argv[1]).read().strip().splitlines()
random.shuffle(files)
tqdm.contrib.concurrent.thread_map(eval_, files, max_workers=4)
