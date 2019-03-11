import ra
import matplotlib.pyplot as plt
import numpy as np

natoms=np.array([10150,20090,32106,45006,67338,78762,91889,111370,138866])
dict_cuda_epg=np.array([1.73,2.58,3.65,4.79,6.80,7.69,8.81,10.5,12.9])
dict_matlab_epg=np.array([543,1060,1910,2390,3950,4270,4889,6030,7610])
match_cuda=np.array([0.54,0.90,1.35,1.81,2.67,3.04,3.51,4.24,5.28])
match_matlab=np.array([8.75,17.4,33.8,37,58.1,73.5,80.8,87,143])

fig, ax = plt.subplots(figsize=(12,9))

fs = 20

ax.semilogy(natoms, dict_cuda_epg, '--', color = 'black', label='Dictionary')
ax.plot(natoms,dict_cuda_epg,'o',color = 'black', label = 'snapMRF',markersize=12)

ax.semilogy(natoms, dict_matlab_epg, '--', color = 'black')
ax.plot(natoms,dict_matlab_epg,'*',color = 'black', markersize=12)

ax.semilogy(natoms, match_cuda, color = 'black')
ax.plot(natoms,match_cuda,'o',color = 'black', markersize=12)

ax.semilogy(natoms, match_matlab, color = 'black',label='Matching')
ax.plot(natoms,match_matlab,'*',color = 'black',label='Ma et al.',markersize=12)

ax.set_xscale("log", nonposx='clip')
ax.set_ylim(0.1,100000)
ax.tick_params(axis='both', which='major', length=10, direction='in')
ax.tick_params(axis='both', which='minor', length=8, direction='in')
ax.set(xlabel='Dictionary Size (atoms)', ylabel='Time (s)')
ax.grid(True)
ax.legend(loc=2,prop={'size': fs})

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fs)

# plt.show()
fig.savefig("../fig/time.eps",bbox_inches='tight')
fig.savefig("../fig/time.png",bbox_inches='tight')
