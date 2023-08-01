import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()
from scipy.optimize import minimize

# def hexedges(Ntop, fibre_diameter, angle=0):
#     L = np.linspace(0,2.*np.pi,7)-angle
#     xv = 0.5*np.cos(L).T*(np.sqrt(3)*(Ntop-1)+1)*fibre_diameter*2/np.sqrt(3)
#     yv = 0.5*np.sin(L).T*(np.sqrt(3)*(Ntop-1)+1)*fibre_diameter*2/np.sqrt(3)
#     return xv, yv
#
#
# Ntop = 5
# Nmiddle = 2 * Ntop-1
# Nfibres = 0.75 * Nmiddle**2 + 0.25
# angle = 0 #-19 #-24
#
# core_diameter = 6.5
# cladding_thickness = 27.3
#
# # excluded = []
# excluded = [52, 60, 9, 53, 61, 1]
#
# fibre_diameter = core_diameter + 2*cladding_thickness
# [ xv, yv ] = hexedges(Ntop, fibre_diameter, 0)
# [gridx,gridy] = hex(fibre_diameter, Ntop, false, angle)
#
# gridx(excluded) = [];
# gridy(excluded) = [];
# Nfibres = length(gridx);
#
#
# figure(1)
# plot(gridx, gridy, 'o')
# axis equal
#
# for k = 1:Nfibres
#     disp(['[' num2str(gridx(k)) ', ' num2str(gridy(k)) '],'])
# end




def hex_center(x, y, z, edge_length):
    return ((1 * x      - 0.5 * y       - 0.5 * z) * edge_length,
            (       np.sqrt(3) / 2 * y - np.sqrt(3) / 2 * z) * edge_length)


def make_hexgrid(hex_range=(-5,5), edge_length=10, grid_rad_px=40, rotate=0, offset=None,
                 aspect=None, excl_inds=None):
    hex_y, hex_x = np.mgrid[hex_range[0]:hex_range[1], hex_range[0]:hex_range[1]]
    y,x = hex_center(hex_y, hex_x, 0, edge_length)
    x = x.ravel()
    y = y.ravel()
    r = np.sqrt(x**2 + y**2)
    ok = r <= grid_rad_px
    x = x[ok]
    y = y[ok]

    if excl_inds is not None:
        try:
            x = np.delete(x, excl_inds)
            y = np.delete(y, excl_inds)
        except:
            pass

    if rotate != 0:
        th = rotate/180*np.pi
        R = np.array([ [np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)] ])
        [y,x] = R @ [y,x]

    if aspect is not None:
        x = x * aspect

    if offset is not None:
        y = y + offset[0]
        x = x + offset[1]

    return y, x


edge_length = 13
grid_rad_px = 4 * edge_length
offset = [79,89]
aspect = 0.75
rotate = 14
p0 = [edge_length, rotate, offset, aspect]

yposn, xposn = make_hexgrid(hex_range=(-5,5), edge_length=edge_length, grid_rad_px=grid_rad_px, rotate=rotate,
                            offset=offset, aspect=aspect, excl_inds=[0, 4, 34, 60, 56, 26])

n_elems = xposn.size
plt.clf()
plt.plot(xposn,yposn,'x')
for k in range(n_elems):
    plt.text(xposn[k], yposn[k], '%d' % (k))
plt.gca().axis('equal')


refim = np.load('../refim.npy')
refim /= np.max(refim)
testim = np.zeros_like(refim)
imY, imX = np.mgrid[0:refim.shape[0], 0:refim.shape[1]]

def make_2dgaussian(y, x, A, fwhm, imY, imX):
    c = fwhm/2.35482
    im_gauss = A * np.exp(-((imY-y)**2/(2*c**2) +  (imX-x)**2/(2*c**2)))
    return im_gauss


ampl = 1
fwhm=5
# im_gauss = make_2dgaussian(gauss_posn[0], gauss_posn[1], ampl, fwhm, imY, imX)
testim = np.zeros_like(refim)
for k in range(n_elems):
    im_gauss = make_2dgaussian(yposn[k], xposn[k], ampl, fwhm, imY, imX)
    testim += im_gauss
plt.clf()
plt.imshow(testim)


plt.clf()
fitim = (refim - testim)**2
# fitim = np.abs(refim - testim)
# fitim = (refim - testim)
gof = np.mean(fitim)
plt.imshow(fitim)
plt.title(gof)



#######################

refim = np.load('../refim.npy')
refim /= np.max(refim)
testim = np.zeros_like(refim)
imY, imX = np.mgrid[0:refim.shape[0], 0:refim.shape[1]]

ampl = 1
fwhm=5
grid_rad_px = 4 * edge_length

edge_length = 13
offset = [79, 89]
aspect = 0.75
rotate = 14
p0 = np.array([edge_length, rotate, offset[0], offset[1], aspect])

def modelfunc(p, showplot=False, returnposns=False):
    edge_length = p[0]
    rotate = p[1]
    offset=[p[2], p[3]]
    aspect = p[4]
    yposn, xposn = make_hexgrid(hex_range=(-5, 5), edge_length=edge_length, grid_rad_px=grid_rad_px, rotate=rotate,
                                offset=offset, aspect=aspect, excl_inds=[0, 4, 34, 60, 56, 26])

    testim = np.zeros_like(refim)
    for k in range(n_elems):
        im_gauss = make_2dgaussian(yposn[k], xposn[k], ampl, fwhm, imY, imX)
        testim += im_gauss

    fitim = (refim - testim) ** 2
    # fitim = np.abs(refim - testim)
    # fitim = (refim - testim)
    gof = np.mean(fitim)

    if showplot:
        plt.clf()
        plt.subplot(211)
        plt.imshow(refim - testim)
        plt.subplot(212)
        plt.imshow(refim)
        plt.plot(xposn, yposn, 'xw')
        for k in range(n_elems):
            plt.text(xposn[k], yposn[k], '%d' % (k), color='white')

    if returnposns:
        return yposn, xposn
    else:
        return gof


res = minimize(modelfunc, p0, options={'Disp': True})
p_opt = res.x

yposn_opt, xposn_opt = modelfunc(p_opt, showplot=True, returnposns=True)

save = False
datadir = './'

if save:
    savefilename = 'plcoords_mono_20230605_01.npz'
    np.savez(datadir+savefilename, p_opt=p_opt, yposn=yposn_opt, xposn=xposn_opt,
             refim=refim)






