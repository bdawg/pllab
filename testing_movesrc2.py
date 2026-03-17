from wip_my_mcc_move import PlanetSimulator
import numpy as np
import time
ps = PlanetSimulator()
ps.set_contrast(1)
# ps.set_contrast(0.3)


xrange = [-0.5, 0.5]
yrange = [-0.5, 0.5]
contr_range = [0.3, 1]
nsteps = 25#100
nreps = 1#00
waittime = 0.2

# X,Y scan
xposns = np.linspace(xrange[0], xrange[1], int(np.sqrt(nsteps)))
yposns = np.linspace(yrange[0], yrange[1], int(np.sqrt(nsteps)))

totsteps = np.shape(xposns)[0] * np.shape(yposns)[0]
for rep in range(nreps):
    print('Rep %d' % rep)
    iter = 0
    for xposn in xposns:
        for yposn in yposns:
            if iter % 10 == 0:
                print('Iter %d of %d' % (iter, totsteps))
            ps.set_position((xposn, yposn))
            iter += 1
            time.sleep(waittime)


# # Random posn
# xposns = np.random.uniform(xrange[0], xrange[1], nsteps)
# yposns = np.random.uniform(yrange[0], yrange[1], nsteps)
# contrs = np.random.uniform(contr_range[0], contr_range[1], nsteps)
#
# for k in range(nsteps):
#     if k % 10 == 0:
#         print('Iter %d of %d' % (k, nsteps))
#     ps.set_position((xposns[k], yposns[k]))
#     ps.set_contrast(contrs[k])
#     k += 1
#     time.sleep(waittime)






