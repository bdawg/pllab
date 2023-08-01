import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import psutil
import os
from sklearn.model_selection import train_test_split
plt.ion()

datapath = '/Users/bnorris/DontBackup/PL/202209_scexaoPL/'
outpath = '/Users/bnorris/DontBackup/PL/202209_scexaoPL/NNoutputs/'

datapath = '/home/bnorris/Data/preprocdata_20220929/'
datapath = '/home/bnorris/Data/preprocdata_20220929/20221016a/'
outpath = '/home/bnorris/Data/NNoutputs_20220929/'


pldata_filename = 'preprocdata_20220929_set07_01_glint_subsetfrms_0-7fracDelay.npz'
coeffs_filename = 'zernLWEdata_20220929_set07_01.npz'

pldata_filename = 'preprocdata_20221016_no-ocam_oldtm_Set05_glint_subsetfrms_0-85fracDelay_no-ocam.npz'
coeffs_filename = 'zernLWEdata_20221016_set05.npz'
pldata_filename = 'preprocdata_20221016_no-ocam_oldtm_Set06_glint_subsetfrms_0-85fracDelay_no-ocam.npz'
coeffs_filename = 'zernLWEdata_20221016_set06.npz'

save_label = ''
save_label = 'current'
# save_label = 'm01'

save_preds = True
save_model = True
model_filename = 'PL-CF_' + pldata_filename[13:-4]  + '_' + save_label +'_model'
outfilename = 'PL-CF_' + pldata_filename[13:-4] + '_' + save_label + '_preddata'

stat_frms = 10000 # How many frames to use for normalisation statistics
testDataSplit = 0.2


pdict = {}
pdict['actFunc'] = 'relu'
# pdict['actFunc'] = tf.keras.layers.LeakyReLU(alpha=0.3) #'relu'
pdict['batchSize'] = 32
pdict['learningRate'] = 0.00001
pdict['lossFunc'] = 'mean_squared_error'
pdict['n_units'] = 2000
pdict['epochs'] = 50
pdict['dropout_rate'] = 0.2
print(pdict)




print("Loading " + pldata_filename)
f = np.load(datapath + pldata_filename)
pl_fluxes = f['all_fluxes']
timingdata = f['timingdata']

print("Loading " + coeffs_filename)
f = np.load(datapath + coeffs_filename)
coeffs_in = f['coeffs']
coeffs_timingdata = f['all_frametimes']
try:
    dmims_in = f['all_frames']
except:
    print('No dm images in zlwe file')

# Normalising PLfluxes
# mn = np.min(pl_fluxes[:stat_frms])
# pl_fluxes -= mn
# mx = np.max(pl_fluxes[:stat_frms])
# print('Normalising PL data by max value (%f)' % mx)
# pl_fluxes /= mx
# pl_fluxes -= np.std(pl_fluxes) # Make zero mean
mean = np.mean(pl_fluxes[:stat_frms])
pl_fluxes -= mean
sd = np.std(pl_fluxes[:stat_frms])
pl_fluxes /= sd


X_train, X_test, y_train, y_test = train_test_split(pl_fluxes, coeffs_in, test_size=testDataSplit, shuffle=True)
# X_train, X_test, y_train, y_test = train_test_split(coeffs_in, pl_fluxes, test_size=testDataSplit, shuffle=True)

Xndims = X_test.shape[1]
yndims = y_test.shape[1]
model = keras.Sequential([
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc'], input_shape = (Xndims,)),
    keras.layers.Dropout(pdict['dropout_rate']),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    keras.layers.Dropout(pdict['dropout_rate']),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    keras.layers.Dropout(pdict['dropout_rate']),
    keras.layers.Dense(yndims)
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=pdict['learningRate']),
              loss=pdict['lossFunc'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=pdict['epochs'], batch_size=pdict['batchSize'])

predictions = model.predict(X_test)
history_loss = history.history['loss']
history_val_loss = history.history['val_loss']

if save_model:
    print('Saving model to ' + outpath + model_filename + '.h5')
    model.save(outpath + model_filename + '.h5', save_format='h5')
    np.savez(outpath + model_filename + '_metadata.npz', pdict=pdict,
             pldata_filename=pldata_filename, loss=history.history['loss'],
             val_loss=history.history['val_loss'])

if save_preds:
    np.savez(outpath + outfilename+'.npz', predvals=predictions, testvals=y_test, history_loss=history_loss,
             history_val_loss=history_val_loss, X_test=X_test, pdict=pdict, pldata_filename=pldata_filename)



# Plot training & validation loss values
plt.figure(1)
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss - val=%.3g' % history.history['val_loss'][-1])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot first n predictions
nplot=100
plt.figure(2, figsize=(9,6))
plt.clf()
for k in range(yndims):
    plt.subplot(yndims,1,k+1)
    plt.plot(y_test[:nplot,k])
    # plt.plot(predictions[:nplot, k])
    plt.plot(predictions[:nplot,k], 'ro', markersize=3, alpha=0.4)


rmse = np.sqrt(np.min(history.history['val_loss']))
wl=1550
print('Equals %f radians RMS, %f nm @ 1550nm (PER TERM)' % (rmse, rmse/(2*np.pi)*wl))
print('Approx. %f radians RMS, %f nm @ 700nm overall' % (rmse*np.sqrt(yndims),
                                                         rmse/(2*np.pi)*700*np.sqrt(yndims)))
av_coeff = np.mean(np.abs(coeffs_in))
print('Applied average %f radians RMS, %f nm @ 1550nm (PER TERM)' % (av_coeff, av_coeff / (2 * np.pi) * wl))
print('Approx %f radians RMS, %f nm @ 700nm overall' % (av_coeff * np.sqrt(yndims),
                                                        av_coeff / (2 * np.pi) * wl * np.sqrt(yndims)))