import sys
import numpy as np
import tensorflow as tf
import librosa
import tqdm
import pickle
import random

if int(sys.argv[1]):
    def load(f, chan):
        r, _ = librosa.core.load(f,sr=16000,mono=False)
        if chan >= 0:
            w = r[chan]
        else:
            w = r
        w = np.asfortranarray(librosa.util.fix_length(w,32000))
        return (w,np.abs(librosa.core.stft(w,1024)))

    X = [load(f, int(chan)) for f, chan, _ in
         tqdm.tqdm([x.split(';') for x in open('rir-small-rooms-n.log').read().splitlines()])] \
      + [load(f, int(chan)) for f, chan, _ in
         tqdm.tqdm([x.split(';') for x in open('rir-small-rooms-y.log').read().splitlines()])]
    Y = [False for _ in open('rir-small-rooms-n.log').read().splitlines()] \
      + [True  for _ in open('rir-small-rooms-y.log').read().splitlines()]
    pickle.dump((X,Y), open('small-rooms-xy.pkl', 'wb'))
else:
    X,Y = pickle.load(open('small-rooms-xy.pkl','rb'))

xy = list(zip(X,Y))
random.shuffle(xy)
X,Y=list(zip(*xy))

if 1:
    X,Y=np.array([x[1] for x in X]),np.array(Y)

    inp = tf.keras.Input(X[0].shape)
    dense1=tf.keras.layers.Dense(300,activation='relu')(tf.keras.layers.Flatten()(inp))
    dense2=tf.keras.layers.Dense(50,activation='relu')(tf.keras.layers.Dropout(0.2)(dense1))
    dense3=tf.keras.layers.Dense(50,activation='relu')(dense2)
    out=tf.keras.layers.Dense(1,activation='sigmoid')(dense3)

    model = tf.keras.Model(inputs=inp,outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),loss='binary_crossentropy', metrics=['acc'])
    model.fit(X,Y,epochs=10,batch_size=5,validation_split=0.3,shuffle=True)

    model.save('small-rooms-magspec.model')
elif 0:
    X,Y=np.array([x[0] for x in X]),np.array(Y)

    inp = tf.keras.Input(X[0].shape)
    dense1=tf.keras.layers.Dense(100,activation='relu')(tf.keras.layers.Flatten()(inp))
    dense2=tf.keras.layers.Dense(50,activation='relu')(tf.keras.layers.Dropout(0.1)(dense1))
    out=tf.keras.layers.Dense(1,activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=inp,outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss='binary_crossentropy', metrics=['acc'])
    model.fit(X,Y,epochs=20,batch_size=5,validation_split=0.3,shuffle=True)

    model.save('small-rooms-waveform.model')
else:
    X,Y=np.array([librosa.feature.rms(x[0], frame_length=1024)[0] for x in X]),np.array(Y)

    inp = tf.keras.Input(X[0].shape)
    dense1=tf.keras.layers.Dense(100,activation='relu')(tf.keras.layers.Flatten()(inp))
    dense2=tf.keras.layers.Dense(10,activation='relu')(tf.keras.layers.Dropout(0.1)(dense1))
    out=tf.keras.layers.Dense(1,activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=inp,outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss='binary_crossentropy', metrics=['acc'])
    model.fit(X,Y,epochs=100,batch_size=5,validation_split=0.3,shuffle=True)

    model.save('small-rooms-rms.model')
