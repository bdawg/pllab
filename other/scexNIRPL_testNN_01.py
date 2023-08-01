import numpy as np
# import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# from matplotlib import gridspec
# import time
# import psutil
import os
from sklearn.model_selection import train_test_split
plt.ion()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi


datapath = './data/'
responsedata_filename = 'PLrespdata_20220217-001_10modes_5XradScale0.7_n100000'

modelsave_filename= None
modelsave_filename = 'modelsave_20220217-001_10modes_5XradScale0.7_n100000-001'

subsetdata = None
# subsetdata = 10000

trimzeros = True
ignore_term0 = True

stat_frms = 10000 # How many frames to use for normalisation statistics
testDataSplit = 0.2

pdict = {}
pdict['actFunc'] = 'relu'
# pdict['actFunc'] = tf.keras.layers.LeakyReLU(alpha=0.3) #'relu'
pdict['batchSize'] = 32
pdict['learningRate'] = 0.0001
pdict['lossFunc'] = 'mean_squared_error'
pdict['n_units'] = 1000
pdict['epochs'] = 100
pdict['dropout_rate'] = 0.2

random_state = 0
nplot = 100

####

respfile = np.load(datapath+responsedata_filename+'.npz', allow_pickle=True)
print('Using respnse data file ' + responsedata_filename)
nmodes = int(respfile['nmodes'])
all_plfluxes = respfile['all_plfluxes'].T
all_coeffs_rad = respfile['all_coeffs_rad']
zero_firstn = respfile['zero_firstn']
wl = respfile['wl']
wfcubefilename = str(respfile['wfcubefile'])
# wffile = np.load(datapath+wfcubefilename+'.npz')
# all_dmmaps = wffile['all_dmmaps']

if trimzeros:
    all_coeffs_rad = all_coeffs_rad[zero_firstn:, :]
    all_plfluxes = all_plfluxes[zero_firstn:, :]
    # all_dmmaps = all_dmmaps[zero_firstn:, :, :]
if subsetdata is not None:
    all_coeffs_rad = all_coeffs_rad[:subsetdata, :]
    all_plfluxes = all_plfluxes[:subsetdata, :]
    # all_dmmaps = all_dmmaps[:subsetdata, :, :]
if ignore_term0:
    all_coeffs_rad = all_coeffs_rad[:,1:]
    nmodes -= 1
ndata = all_coeffs_rad.shape[0]

# Normalising PLfluxes
mn = np.min(all_plfluxes[:stat_frms, :])
all_plfluxes -= mn
mx = np.max(all_plfluxes[:stat_frms, :])
all_plfluxes /= mx
sd = np.std(all_plfluxes[:stat_frms, :])
all_plfluxes -=  sd # Make zero mean  ######### Mistake? Why subtract sd, not mean?
normfacts = np.array([mn, mx, sd])

X_train, X_test, y_train, y_test = train_test_split(all_plfluxes, all_coeffs_rad, test_size=testDataSplit,
                                                    shuffle=True, random_state=random_state)

Xndims = X_test.shape[1]
yndims = y_test.shape[1]

model = keras.Sequential([
    # keras.layers.Flatten(input_shape = (n_intypes,1)),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc'], input_shape = (Xndims,)),
    keras.layers.Dropout(pdict['dropout_rate']),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    keras.layers.Dropout(pdict['dropout_rate']),
    # keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    # keras.layers.Dropout(pdict['dropout_rate']),
    # keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    keras.layers.Dense(yndims)
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=pdict['learningRate']),
              loss=pdict['lossFunc'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=pdict['epochs'], batch_size=pdict['batchSize'])
print('Fit done.')

predictions = model.predict(X_test)
history_loss = history.history['loss']
history_val_loss = history.history['val_loss']

if modelsave_filename is not None:
    print('Saving model as ' + modelsave_filename)
    model.save(datapath + modelsave_filename + '.h5', save_format='h5')
    np.savez(datapath + modelsave_filename + '.npz', pdict=pdict, responsedata_filename=responsedata_filename,
             history_loss=history_loss, history_val_loss=history_val_loss, predictions=predictions,
             ndata=ndata, normfacts=normfacts)

# Plot training & validation loss values
plt.figure(1)
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss - val=%.4g' % history.history['val_loss'][-1])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot predictions
plt.figure(2, figsize=[8, 10])
plt.clf()
for k in range(nmodes):
    plt.subplot(nmodes, 1, k + 1)
    plt.plot(y_test[:nplot, k])
    plt.plot(predictions[:nplot, k], '.')
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    # if show_modenames:
    #     plt.text(0.01, 0.8, modeNames[k], transform=ax.transAxes, color='r')
    # plt.title(modeNames[k])
    # plt.text(0.01, 0.1, ('MSE: %.4f' % allMSEs[k]), transform=ax.transAxes, color='b')
    # if xlims is not None:
    #     plt.xlim(xlims)
    # if ylims is not None:
    #     plt.ylim(ylims)
plt.tight_layout()
