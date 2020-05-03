import sys
import numpy as np
import tensorflow as tf
import librosa
import tqdm
import pickle

if int(sys.argv[1]):
    def load(f, chan):
        r, _ = librosa.core.load(f,sr=16000,mono=False)
        if chan >= 0:
            w = r[chan]
        else:
            w = r
        return np.abs(librosa.core.stft(np.asfortranarray(librosa.util.fix_length(w,32000)),1024))

    X = [
        load(f, int(chan))
        for f, chan, _ in
        tqdm.tqdm([x.split(';') for x in open('rir-small-rooms.log').read().splitlines()])
        ]
    Y = [
        1 if y == 'y' else 0
        for _, _, y in
        [x.split(';') for x in open('rir-small-rooms.log').read().splitlines()]
        ]
    pickle.dump((X,Y), open('small-rooms-xy.pkl', 'wb'))
else:
    X,Y = pickle.load(open('small-rooms-xy.pkl','rb'))

X,Y=np.array(X),np.array(Y)

inp = tf.keras.Input(X[0].shape)
dense1=tf.keras.layers.Dense(500,activation='relu')(tf.keras.layers.Flatten()(inp))
dense2=tf.keras.layers.Dense(100,activation='relu')(dense1)
dense3=tf.keras.layers.Dense(100,activation='relu')(dense2)
out=tf.keras.layers.Dense(1,activation='sigmoid')(dense3)

model = tf.keras.Model(inputs=inp,outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),loss='binary_crossentropy', metrics=['acc'])
model.fit(X,Y,epochs=10,batch_size=5,validation_split=0.3)

model.save('small-rooms.model')
