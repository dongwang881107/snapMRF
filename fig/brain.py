import ra
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(15,5))
fs = 13
mask = ra.read('../fig/mask_brain.ra')

path_epg = 'maps500k_brain_varTR_b1map'
maps_epg = ra.read('../result/' + path_epg + '.ra')

t1_epg = np.reshape(maps_epg[0,:],(240,240),order='F')
t2_epg = np.reshape(maps_epg[1,:],(240,240),order='F')
b0_epg = np.reshape(maps_epg[2,:],(240,240),order='F')
b1_epg = np.reshape(maps_epg[3,:],(240,240),order='F')
pd_epg = np.reshape(maps_epg[4,:],(240,240),order='F')
pd_epg = pd_epg/np.max(pd_epg[:])

t1_epg = t1_epg*mask
t2_epg = t2_epg*mask
b0_epg = b0_epg*mask
b1_epg = b1_epg*mask

plt.subplot(241)
plt.imshow(t1_epg,aspect='equal',cmap='inferno')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(t1_epg[:]),np.max(t1_epg[:]),5))
plt.xticks([])
plt.yticks([])
plt.ylabel('snapMRF',fontsize=fs)
plt.title(r"$T_1$ (ms)",fontsize=fs)

plt.subplot(242)
plt.imshow(t2_epg,aspect='equal',cmap='inferno', vmax=300)
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0,300,4))
plt.axis('off')
plt.title(r"$T_2$ (ms)",fontsize=fs)

plt.subplot(243)
plt.imshow(b0_epg,aspect='equal',cmap='PuOr')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(b0_epg[:]),np.max(b0_epg[:]),5))
plt.axis('off')
plt.title(r"off-resonance (Hz)",fontsize=fs)

plt.subplot(244)
plt.imshow(pd_epg,aspect='equal',cmap='inferno', vmax=0.25)
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0,0.25,6))

plt.axis('off')
plt.title(r"proton density",fontsize=fs)

path_roa = 'maps500k_brain_varTR_b1map_matlab'
maps_roa = ra.read('../result/' + path_roa + '.ra')

t1_roa = np.reshape(maps_roa[0,:],(240,240),order='F')
t2_roa = np.reshape(maps_roa[1,:],(240,240),order='F')
b0_roa = np.reshape(maps_roa[2,:],(240,240),order='F')
b1_roa = np.reshape(maps_roa[3,:],(240,240),order='F')
pd_roa = np.reshape(maps_roa[4,:],(240,240),order='F')
pd_roa = pd_roa/np.max(pd_roa[:])

t1_roa = t1_roa*mask
t2_roa = t2_roa*mask
b0_roa = b0_roa*mask
b1_roa = b1_roa*mask

plt.subplot(245)
plt.imshow(t1_roa,aspect='equal',cmap='inferno')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(t1_roa[:]),np.max(t1_roa[:]),5))
plt.xticks([])
plt.yticks([])
plt.ylabel('Ma et al.',fontsize=fs)

plt.subplot(246)
plt.imshow(t2_roa,aspect='equal',cmap='inferno', vmax=300)
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0,300,4))
plt.axis('off')

plt.subplot(247)
plt.imshow(b0_roa,aspect='equal',cmap='PuOr')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(b0_roa[:]),np.max(b0_roa[:]),5))
plt.axis('off')

plt.subplot(248)
plt.imshow(pd_roa,aspect='equal',cmap='inferno', vmax = 0.25)
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0,0.25,6));
plt.axis('off')

# plt.show()
path = 'brain_varTR'
fig.savefig('../fig/' + path + '.eps',bbox_inches='tight')
fig.savefig('../fig/' + path + '.png',bbox_inches='tight')
