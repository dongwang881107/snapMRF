import ra
import matplotlib.pyplot as plt
import numpy as np

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    circle = dist_from_center <= radius
    return circle

radius = 64
circle = create_circular_mask(240,240,[120,120],radius)

path = 'maps400k_phantom_varTR_b1map'

img = ra.read('../fig/data_phantom.ra')
maps = ra.read('../result/' + path + '.ra')
mask = ra.read('../fig/mask_phantom.ra').astype('float')

fs = 13

fig, ax = plt.subplots(figsize=(15,7.5))

img = abs(img)

plt.subplot(231)
plt.imshow(img,aspect='equal',cmap='inferno')
plt.axis('off')
plt.title('Raw Image',fontsize=fs)

t1 = np.reshape(maps[0,:],(240,240),order='F')
t2 = np.reshape(maps[1,:],(240,240),order='F')
b0 = np.reshape(maps[2,:],(240,240),order='F')
b1 = np.reshape(maps[3,:],(240,240),order='F')
pd = np.reshape(maps[4,:],(240,240),order='F')

t1[t1>=2600]=0
t2[t2>=900]=0

t1 = t1*circle
t2 = t2*circle
mask1 = (t1>0)

t1 = t1*mask1
t2 = t2*mask1

b0 = b0*mask
b1 = b1*mask
pd = pd*mask

b1[b1==0]=1

pd = pd/np.max(pd[:])

plt.subplot(232)
plt.imshow(t1,aspect='equal',cmap='inferno')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(t1[:]),np.max(t1[:]),6))

plt.axis('off')
plt.title(r"$T_1$ (ms)",fontsize=fs)

plt.subplot(233)
plt.imshow(t2,aspect='equal',cmap='inferno')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(t2[:]),np.max(t2[:]),5))
plt.axis('off')
plt.title(r"$T_2$ (ms)",fontsize=fs)

plt.subplot(234)
plt.imshow(b1,aspect='equal',cmap='PuOr',vmin=0.5,vmax=1.5)
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0.5,1.5,3))
plt.axis('off')
plt.title(r"$B_1^+$",fontsize=fs)

plt.subplot(235)
plt.imshow(b0,aspect='equal',cmap='PuOr')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(b0[:]),np.max(b0[:]),7))
plt.axis('off')
plt.title(r"off-resonance (Hz)",fontsize=fs)

plt.subplot(236)
plt.imshow(pd,aspect='equal',cmap='inferno')
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(np.min(pd[:]),np.max(pd[:]),6))
plt.axis('off')
plt.title(r"proton density",fontsize=fs)

# plt.show()
path = 'phantom_varTR'
fig.savefig('../fig/' + path + '.eps',bbox_inches='tight')
