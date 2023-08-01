# import matplotlib
# matplotlib.use('TkAgg')

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

datapath = '/Users/bnorris/DontBackup/PL/preprocdata_20220324/'
# datapath = '/home/bnorris/Data/preprocdata_20220324/'

outpath = '/Users/bnorris/DontBackup/PL/outputs_20220324/'
# outpath = '/media/morgana2/barnaby/PL/outputs_20220324/'

pldata_filename = 'preprocdata_20220324_1430_smth7_500sec_10Kfrms_glint.npz'
# pldata_filename = 'preprocdata_20220324_1430_smth7_500sec_100Kfrms_glint.npz'

wfdata_filename = 'preprocdata_20220324_1430_smth7_500sec_10Kfrms_ocam2d_WAVEFRONT_nmodes20.npz'
# wfdata_filename = 'preprocdata_20220324_1430_smth7_500sec_100Kfrms_ocam2d_WAVEFRONT_nmodes20.npz'
mask_wf = False

save_label = ''
save_label = 'current'
# save_label = 'm-id01'

save_preds = True
# outfilename = 'PL-WF_preddata'
outfilename = 'PL-WF_' + pldata_filename[13:-4] + '_' + save_label + '_preddata'

max_nsave = None
max_nsave = 1000 # Only save this many predictions

res_testframes = None
# res_testframes = 1000 # Reserve some contiguous frames from set for later visualisation

use_subset = None #100 # None to use all data

save_model = True
model_filename = 'PL-WF_' + pldata_filename[13:-4]  + '_' + save_label +'_model'

stat_frms = 10000 # How many frames to use for normalisation statistics
testDataSplit = 0.2

pdict = {}
pdict['actFunc'] = 'relu'
# pdict['actFunc'] = tf.keras.layers.LeakyReLU(alpha=0.3) #'relu'
pdict['batchSize'] = 32
pdict['learningRate'] = 0.0001*10
pdict['lossFunc'] = 'mean_squared_error'
pdict['n_units'] = 1000*4
pdict['epochs'] = 100
pdict['dropout_rate'] = 0.0
print(pdict)

random_state = 0



#####

process = psutil.Process(os.getpid())
def showmem():
    mem_info = process.memory_info()
    vms = mem_info.vms / 1024 ** 3
    rss = mem_info.rss / 1024 ** 3
    print('---> Mem (GB) res, virt: %.2f, %.2f' % (rss, vms))

f = np.load(datapath + pldata_filename)
pl_fluxes = f['all_fluxes']
timingdata = f['timingdata']

f = np.load(datapath + wfdata_filename)
wfcube = f['wfcube']
if mask_wf:
    dmmask = f['dmmask']
    dmmaskb = ~dmmask.astype('bool')
    dmmaskb = np.broadcast_to(dmmaskb, wfcube.shape)
    wfcube = np.ma.masked_array(wfcube, mask=dmmaskb)


showmem()
# Normalising PLfluxes
mn = np.min(pl_fluxes[:stat_frms])
pl_fluxes -= mn
mx = np.max(pl_fluxes[:stat_frms])
print('Normalising PL data by max value (%f)' % mx)
pl_fluxes /= mx
pl_fluxes -= np.std(pl_fluxes) # Make zero mean

# plt.clf()
# h=plt.hist(pl_fluxes.ravel(), 100)
showmem()


ndata = wfcube.shape[0]
y_dims = wfcube.shape[1:3]
wfcube_flat = np.reshape(wfcube,(ndata,-1))
wfcube_flat /= np.std(wfcube_flat[:stat_frms,:])
showmem()


if res_testframes is not None:
    n_testdata = int(res_testframes)
    n_traindata = int(ndata - n_testdata)
    splitinds = [int(ndata/2 - n_testdata/2), int(ndata/2 + n_testdata/2)]
    X_test_res = pl_fluxes[splitinds[0]:splitinds[1], :]
    y_test_res = wfcube_flat[splitinds[0]:splitinds[1], :]
    traindata_X_1 = pl_fluxes[:splitinds[0], :]
    traindata_y_1 = wfcube_flat[:splitinds[0], :]
    traindata_X_2 = pl_fluxes[splitinds[1]:, :]
    traindata_y_2 = wfcube_flat[splitinds[1]:, :]
    pl_fluxes = np.vstack((traindata_X_1, traindata_X_2))
    wfcube_flat = np.vstack((traindata_y_1, traindata_y_2))

# Split test data normal way
X_train, X_test, y_train, y_test = train_test_split(pl_fluxes, wfcube_flat, test_size=testDataSplit, shuffle=True,
                                                    random_state=random_state)

if use_subset is not None:
    X_train = X_train[:use_subset, :]
    X_test = X_test[:use_subset, :]
    y_train = y_train[:use_subset, :]
    y_test = y_test[:use_subset, :]



showmem()

Xndims = X_test.shape[1]
yndims = y_test.shape[1]
model = keras.Sequential([
    # keras.layers.Flatten(input_shape = (n_intypes,1)),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc'], input_shape = (Xndims,)),
    keras.layers.Dropout(pdict['dropout_rate']),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    keras.layers.Dropout(pdict['dropout_rate']),
    keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    keras.layers.Dropout(pdict['dropout_rate']),

    # keras.layers.Dense(pdict['n_units'], activation=pdict['actFunc']),
    # keras.layers.Dropout(pdict['dropout_rate']),
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

if save_model:
    print('Saving model to ' + outpath + model_filename + '.h5')
    model.save(outpath + model_filename + '.h5', save_format='h5')
    np.savez(outpath + model_filename + '_metadata.npz', pdict=pdict,
             pldata_filename=pldata_filename, loss=history.history['loss'],
             val_loss=history.history['val_loss'], random_state=random_state)

# # Hack - predict on training data
# X_test = X_train
# y_test = y_train
# print('WARNING!!! Predicting on training data')

predictions = model.predict(X_test)
pred_ims = np.reshape(predictions, (-1,y_dims[0],y_dims[1]))
test_ims = np.reshape(y_test, (-1,y_dims[0],y_dims[1]))
history_loss = history.history['loss']
history_val_loss = history.history['val_loss']

if res_testframes is not None:
    predictions_res = model.predict(X_test_res)
    pred_ims_res = np.reshape(predictions_res, (-1, y_dims[0], y_dims[1]))
    test_ims_res = np.reshape(y_test_res, (-1, y_dims[0], y_dims[1]))
else:
    pred_ims_res = None
    test_ims_res = None
    X_test_res = None

# Save results?
if save_preds:
    if max_nsave is not None:
        pred_ims_save = pred_ims[:max_nsave, :, :]
        test_ims_save = test_ims[:max_nsave, :, :]
        X_test_save = X_test[:max_nsave, :]
    else:
        pred_ims_save = pred_ims
        test_ims_save = test_ims
        X_test_save = X_test
    np.savez(outpath + outfilename+'.npz', pred_ims=pred_ims_save, test_ims=test_ims_save, history_loss=history_loss,
             history_val_loss=history_val_loss, X_test=X_test_save, pdict=pdict, pldata_filename=pldata_filename,
             pred_ims_res=pred_ims_res, test_ims_res=test_ims_res, X_test_res=X_test_res)


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


nshow = 1000
nshow = 1
# nshow = test_ims.shape[0]
cmap = 'viridis'
# clim = None
clim=[-3,3]

# test_ims = test_ims_res
# pred_ims = pred_ims_res

def quickplot(ind=None):
    if ind is None:
        ns = range(nshow)
    else:
        ns = [ind]
    for k in ns:
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(test_ims[k, :, :], cmap=cmap, clim=clim)
        plt.title('True WF')
        plt.subplot(1, 2, 2)
        plt.imshow(pred_ims[k, :, :], cmap=cmap, clim=clim)
        plt.title('Predicted WF')
        plt.pause(0.01)

model.summary()
print(pdict)

# nshow = test_ims.shape[0]
plt.figure(2)
quickplot()

