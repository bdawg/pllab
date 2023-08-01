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

pldata_filename = 'preprocdata_20220324_1430_smth7_500sec_100Kfrms_glint.npz'
ocamdata_filename = 'preprocdata_20220324_1430_smth7_500sec_100Kfrms_ocam2d.npz'
ircamdata_filename = 'preprocdata_20220324_1430_smth7_500sec_100Kfrms_ircam0.npz'

datapath = '/home/bnorris/Data/preprocdata_20220929/'
datapath = '/Users/bnorris/Dropbox/code/PL_NN/sel_outs/'
# pldata_filename = 'preprocdata_20220929_set10_01_glint_subsetfrms_0-7fracDelay.npz'
# pldata_filename = 'preprocdata_20220929_set10_01_glint.npz'
pldata_filename = 'preprocdata_20221016_no-ocam_oldtm_Set06_glint_subsetfrms_0-85fracDelay_no-ocam.npz'
ocamdata_filename = None
# ircamdata_filename = 'preprocdata_20220929_set10_01_ircam0_subsetfrms_0-7fracDelay.npz'
# ircamdata_filename = 'preprocdata_20220929_set10_01_ircam0.npz'
ircamdata_filename = 'preprocdata_20221016_no-ocam_oldtm_Set06_ircam0_subsetfrms_0-85fracDelay_no-ocam.npz'
outpath = './outputs/'


save_label = ''
# save_label = '100epoch_do0-4_resStart'
# save_label = 'scexaobench_20221016_Set05'

save_preds = True
# outfilename = 'PL-PSF_preddata'
outfilename = 'PL-PSF_' + pldata_filename[13:-4] + '_' + save_label + '_preddata'

max_nsave = None
# max_nsave = 1000 # Only save this many predictions

res_testframes = None
res_testframes = 3000 #1000 # Reserve some contiguous frames from set for later visualisation

use_subset = None #100 # None to use all data

save_model = True
model_filename = 'PL-PSF_' + pldata_filename[13:-4]  + '_' + save_label +'_model'

stat_frms = 10000 # How many frames to use for normalisation statistics
testDataSplit = 0.1 #0.2

pdict = {}
pdict['actFunc'] = 'relu'
# pdict['actFunc'] = tf.keras.layers.LeakyReLU(alpha=0.3) #'relu'
pdict['batchSize'] = 32
pdict['learningRate'] = 0.0001
pdict['lossFunc'] = 'mean_squared_error'
pdict['n_units'] = 1000
pdict['epochs'] = 100
pdict['dropout_rate'] = 0.4 #0.4

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
# f = np.load(datapath + ocamdata_filename)
# ocam_cube = f['cube']
f = np.load(datapath + ircamdata_filename)
ircam_cube = f['cube']


showmem()
# Normalising PLfluxes
mn = np.min(pl_fluxes[:stat_frms])
pl_fluxes -= mn
mx = np.max(pl_fluxes[:stat_frms])
print('Normalising PL data by max value (%f)' % mx)
pl_fluxes /= mx
# plt.clf()
# h=plt.hist(pl_fluxes.ravel(), 100)
showmem()

# Normalising ircam0
# topcut_perc = 0.1
# vals = ircam_cube.ravel()
# topcut = np.percentile(vals, (1-topcut_perc))
# vals = vals[vals < (100-topcut_perc)]
mn = np.min(ircam_cube[:stat_frms, :, :])
ircam_cube -= mn
mx = np.max(ircam_cube)
print('Normalising ircam data by max value (%f)' % mx)
ircam_cube /= mx
# plt.clf()
# h=plt.hist(ircam_cube.ravel(), 1000)
showmem()

# Test pl->ircam
ndata = ircam_cube.shape[0]
y_dims = ircam_cube.shape[1:3]
ircam_cube_flat = np.reshape(ircam_cube,(ndata,-1))
showmem()


if res_testframes is not None:
    # n_testdata = int(res_testframes)
    # n_traindata = int(ndata - n_testdata)
    # splitinds = [int(ndata/2 - n_testdata/2), int(ndata/2 + n_testdata/2)]
    # X_test_res = pl_fluxes[splitinds[0]:splitinds[1], :]
    # y_test_res = ircam_cube_flat[splitinds[0]:splitinds[1], :]
    # traindata_X_1 = pl_fluxes[:splitinds[0], :]
    # traindata_y_1 = ircam_cube_flat[:splitinds[0], :]
    # traindata_X_2 = pl_fluxes[splitinds[1]:, :]
    # traindata_y_2 = ircam_cube_flat[splitinds[1]:, :]
    # pl_fluxes = np.vstack((traindata_X_1, traindata_X_2))
    # ircam_cube_flat = np.vstack((traindata_y_1, traindata_y_2))
    n_testdata = int(res_testframes)
    n_traindata = int(ndata - n_testdata)
    splitinds = [0, int(n_testdata)]
    X_test_res = pl_fluxes[splitinds[0]:splitinds[1], :]
    y_test_res = ircam_cube_flat[splitinds[0]:splitinds[1], :]
    traindata_X_1 = pl_fluxes[:splitinds[0], :]
    traindata_y_1 = ircam_cube_flat[:splitinds[0], :]
    traindata_X_2 = pl_fluxes[splitinds[1]:, :]
    traindata_y_2 = ircam_cube_flat[splitinds[1]:, :]
    pl_fluxes = np.vstack((traindata_X_1, traindata_X_2))
    ircam_cube_flat = np.vstack((traindata_y_1, traindata_y_2))

# Split test data normal way
X_train, X_test, y_train, y_test = train_test_split(pl_fluxes, ircam_cube_flat, test_size=testDataSplit, shuffle=True,
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


# nshow = 1000
nshow = 100
cmap = 'inferno'
clim = [-1.65, 0]
clim = [-1.75, 0]
clim = [-1.6, -0.2]

clim=[-1.5,-0.1]
# clim=[-1.5,-1]
cmap='viridis'
clim=[-2.5,-2.2]
clim=None
#
# test_ims = test_ims_res
# pred_ims = pred_ims_res

# bs = np.min([test_ims.ravel(), pred_ims.ravel()]) # Get bias to avoid -ive pixels for viewing
bs = 0
def quickplot(ind=None):
    if ind is None:
        ns = range(nshow)
    else:
        ns = [ind]
    for k in ns:
        plt.clf()
        plt.subplot(1,2,1)
        # plt.imshow(np.sqrt(test_ims[k,:,:] - bs), cmap=cmap)
        plt.imshow(np.log10(test_ims[k,:,:] - bs), cmap=cmap, clim=clim)
        plt.title('True image')
        plt.subplot(1,2,2)
        # plt.imshow(np.sqrt(pred_ims[k,:,:] - bs), cmap=cmap)
        plt.imshow(np.log10(pred_ims[k,:,:] - bs), cmap=cmap, clim=clim)
        plt.title('Predicted image')
        # if k == 0:
        if clim is not None:
            plt.text(0, 3, 'vmin=%.2f' % clim[0], color='w', fontsize=8)
        plt.pause(0.1)



plt.figure(2)
quickplot()