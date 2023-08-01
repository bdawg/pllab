from multiprocessing import Pool
import time
import numpy as np
import matplotlib.pyplot as plt
from sim_inject_complspeck import multi_complspeckles


def makedatafunc(idnum):
    mfhx_px = 200

    npx_pup = 512
    pupildiam_px = 256
    padding_px = 256
    cropsize = 64  # For plotting
    inp_pix_scale = 8  # input pixels / fiber-field pixels

    # npx_pup = 512 //2
    # pupildiam_px = 256 //2
    # padding_px = 256 //4
    # cropsize = 48
    # inp_pix_scale = 10  # input pixels / fiber-field pixels

    # # ~input of oversampled PL
    wavelength = 1.55 # microns
    core_radius = 39.4/2 # microns

    singlespeck = False
    random_seed = None

    dc_power_range = [0.1, 1]
    cycles_per_pupil_range = [1, 2.5]
    ampl1_range = [0.1, 1]
    ampl2_range = [0.1, 1]
    rot_range = [0, 180]
    phi_range = [0, 100]

    # Try single-speckle mode
    singlespeck = True
    cycles_per_pupil_range = [0, 2.5]
    ampl1_range = [0.1, 1]
    rot_range = [-180, 180]
    phi_range = [0, 100]

    num_data = 2000
    num_data = 1562 # For 64, 8 or 4
    # num_data = 1786  # For 56, 7
    # num_data = 1667  # For 60, 6
    # num_data = 1667  # For 60, 6
    nspecks = 2
    showplots = False
    make_slmims = False

    savedir = '../pllab_data/'
    savedir = '/Users/bnorris/DontBackup/PL/'
    # savedir = '/media/data/bnorris/pl_simdata/'
    savedir = '/home/bnorris/Data/PL/simdata/'

    savefilepref = 'siminjout_2ssp_ampl01-1_20230801b-01_%.2d.npz'
    # savefilepref = 'testout2_%.2d.npz'

    print('Starting set no. %d' % idnum)
    savefile = savefilepref % idnum

    spk = multi_complspeckles(npx_pup, pupildiam_px, padding_px, cropsize, randomseed=random_seed, idnum=idnum)
    spk.setup_fibermodes(core_radius=core_radius, wavelength=wavelength, plot_modefields=False, mfhx_px=mfhx_px)
    spk.make_multi_simdata(num_data, nspecks, dc_power_range, cycles_per_pupil_range, ampl1_range, ampl2_range,
                           rot_range, phi_range, inp_pix_scale, only_one_dc=True, showplots=showplots,
                           savefile=savefile, savedir=savedir, make_slmims=make_slmims, singlespeck=singlespeck,
                           save_reduced=True)


if __name__ == '__main__':
    numfiles = 50 #64
    workers = 2 #4

    idnums = range(numfiles)
    # idnums = range(50,100)
    with Pool(workers) as p:
        p.map(makedatafunc, idnums, chunksize=1)







