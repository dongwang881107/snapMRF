# snapMRF: GPU-Accelerated Magnetic Resonance Fingerprinting Dictionary Generation and Matching using Extended Phase Graphs

snapMRF is an open-source CUDA-based GPU code to generate MRF dictionaries and parameter maps as
fast as possible with validated accuracy. 

Dictionaries can be generated using both Bloch equation simulation (ROA) and extended phase graphs (EPG), and parameter maps are reconstructed using template matching (maximum inner product). 

<!--![brain](https://github.com/chixindebaoyu/snapMRF/raw/master/fig/brain_varTR.png "Example Reconstruction") -->
<img src="https://github.com/chixindebaoyu/snapMRF/raw/master/fig/brain_varTR.png" alt="Example Reconstruction" width="800"/>

Top row: Example in vivo brain parameter maps generated using snapMRF. From left to right: T1,
T2, off-resonance, and proton density, respectively. Bottom row: Parameter
maps generated using the ROA-based MATLAB code of Ma et al.

<!-- ![timing](https://github.com/chixindebaoyu/snapMRF/raw/master/fig/time.png "Timing Results") -->
<img src="https://github.com/chixindebaoyu/snapMRF/raw/master/fig/time.png" alt="Timing Results" width="600"/>

Run time comparison between snapMRF and MATLAB in dictionary generation and
matching. Note the log scale. Time increases linearly with the dictionary size,
showing efficient parallelization. For this example, with 240 x 240 image
voxels, matching took much less time than dictionary generation.



## Dependencies

This code has been tested using 
- Ubuntu 18.04
- CUDA 10.0
- Python 3.5

At minimum a working CUDA installation, including cuBLAS, is required to run the 
code. Python is required to generate the figures from the paper.

## Installing

To install, run the following commands in the `src/` subdirectory:
- `make`
- `sudo make install`

## Directory Structure
- `/src` Contains CUDA utility functions that can be used to generate MRF dictionaries, reconstruct parameter maps and run unit tests.
- `/matlab` Contains MATLAB utility functions as the comparsion with CUDA.
- `/data` Constains datasets for mrf and unit tests
- `/result` Contains the maps used in the paper.
- `/fig` Contains the python functions that can be used to plot the figures in the paper.

## Usage
```
Usage: mrf [OPTION] <rfpulse.csv> <imgstack.ra>
	-M, --b1map <rafile>		 B1 map input RA file
	-N, --NATOMS			 number of atoms
	-a, --atoms <rafile>		 dictionary atoms output RA file
	-m, --maps  <rafile>		 parameter map output RA file
	-p, --params <rafile>		 dictionary parameters
	-t, --T1 start:step:end		 T1 grid
	-s, --T2 start:step:end		 T2 grid
	-b, --B0 start:step:end		 B0 grid
	-r, --B1 start:step:end		 B1 grid
	-w, --states n			 number of historical states to track
	-e, --echo_type n		 echo type
	-T, --timepoints n		 number of time points
	-G, --gridsize n		 set GPU gridsize
	-B, --blocksize n		 set GPU blocksize
	-h				 show this help
```

## Unit Tests
Run `make` and `./test`
 ```
Begin unit test for epg_se
T1 = 600:600:600
T2 = 100:100:100
B0 = 0:1:0
B1 = 1
l_t1: 1, l_t2: 1, l_b0: 1, l_b1: 1
nreps: 1, NATOMS: 1, natoms: 1
Constructing dictionary...
The 1 th atom
atoms_epg_se[0] = 0.606531 + -0.000000im, atoms_gt[0] = 0.606531 + 0.000000im
epg_se unit test passed!

Begin unit test for epg_sr
T1 = 600
T2 = 100
B0 = 0
B1 = 1
l_t1: 1, l_t2: 1, l_b0: 1, l_b1: 1
nreps: 10, NATOMS: 1, natoms: 1
Constructing dictionary...
The 1 th atom
atoms_epg_sr[0] = 0.857408 + 0.000000im, atoms_gt[0] = 0.857408 + 0.000000im
atoms_epg_sr[1] = 0.673983 + 0.000000im, atoms_gt[1] = 0.673983 + 0.000000im
atoms_epg_sr[2] = 0.630996 + 0.000000im, atoms_gt[2] = 0.630996 + 0.000000im
atoms_epg_sr[3] = 0.622047 + 0.000000im, atoms_gt[3] = 0.622047 + 0.000000im
atoms_epg_sr[4] = 0.620198 + 0.000000im, atoms_gt[4] = 0.620198 + 0.000000im
atoms_epg_sr[5] = 0.619817 + 0.000000im, atoms_gt[5] = 0.619817 + 0.000000im
atoms_epg_sr[6] = 0.619738 + 0.000000im, atoms_gt[6] = 0.619738 + 0.000000im
atoms_epg_sr[7] = 0.619721 + 0.000000im, atoms_gt[7] = 0.619721 + 0.000000im
atoms_epg_sr[8] = 0.619718 + 0.000000im, atoms_gt[8] = 0.619718 + 0.000000im
atoms_epg_sr[9] = 0.619717 + 0.000000im, atoms_gt[9] = 0.619717 + 0.000000im
epg_sr unit test passed!

Begin unit test for epg_tse1
T1 = 1000:1000:1000
T2 = 100:100:100
B0 = 0:1:0
B1 = 1
l_t1: 1, l_t2: 1, l_b0: 1, l_b1: 1
nreps: 3, NATOMS: 1, natoms: 1
Constructing dictionary...
The 1 th atom
atoms_epg_tse1[0] = 0.750000 + -0.000000im, atoms_gt[0] = 0.750000 + -0.000000im
atoms_epg_tse1[1] = 0.937500 + -0.000000im, atoms_gt[1] = 0.937500 + -0.000000im
atoms_epg_tse1[2] = 0.843750 + -0.000000im, atoms_gt[2] = 0.843750 + -0.000000im
epg_tse1 unit test passed!

Begin unit test for epg_tse2
T1 = 600:600:600
T2 = 100:100:100
B0 = 0:1:0
B1 = 1
l_t1: 1, l_t2: 1, l_b0: 1, l_b1: 1
nreps: 10, NATOMS: 1, natoms: 1
Constructing dictionary...
The 1 th atom
atoms_epg_tse2[0] = 0.606531 + -0.000000im, atoms_gt[0] = 0.606531 + -0.000000im
atoms_epg_tse2[1] = 0.367879 + -0.000000im, atoms_gt[1] = 0.367879 + -0.000000im
atoms_epg_tse2[2] = 0.223130 + -0.000000im, atoms_gt[2] = 0.223130 + -0.000000im
atoms_epg_tse2[3] = 0.135335 + -0.000000im, atoms_gt[3] = 0.135335 + -0.000000im
atoms_epg_tse2[4] = 0.082085 + -0.000000im, atoms_gt[4] = 0.082085 + -0.000000im
atoms_epg_tse2[5] = 0.049787 + -0.000000im, atoms_gt[5] = 0.049787 + -0.000000im
atoms_epg_tse2[6] = 0.030197 + -0.000000im, atoms_gt[6] = 0.030197 + -0.000000im
atoms_epg_tse2[7] = 0.018316 + -0.000000im, atoms_gt[7] = 0.018316 + -0.000000im
atoms_epg_tse2[8] = 0.011109 + -0.000000im, atoms_gt[8] = 0.011109 + -0.000000im
atoms_epg_tse2[9] = 0.006738 + -0.000000im, atoms_gt[9] = 0.006738 + -0.000000im
epg_tse2 unit test passed!

Begin unit test for epg_fisp
T1 = 1000:1000
T2 = 100:100
B0 = 0:0
B1 = 1
l_t1: 1, l_t2: 1, l_b0: 1, l_b1: 1
nreps: 3, NATOMS: 1, natoms: 1
Constructing dictionary...
The 1 th atom
atoms_epg_fisp[0] = 0.000000 + -0.475615im, atoms_gt[0] = 0.000000 + -0.475615im
atoms_epg_fisp[1] = 0.000000 + -0.412528im, atoms_gt[1] = 0.000000 + -0.412528im
atoms_epg_fisp[2] = 0.000000 + -0.335848im, atoms_gt[2] = 0.000000 + -0.335848im
epg_fisp unit test passed!

Reading ../data/MRF_5.csv
Begin unit test for epg_ssfp
T1 = 100:100:200
T2 = 20+20:20:40+1000
B0 = 0
B1 = 1
l_t1: 2, l_t2: 3, l_b0: 1, l_b1: 1
nreps: 5, NATOMS: 4, natoms: 4
Constructing dictionary...
The 1 th atom
atoms_epg_ssfp[0] = 0.000000 + 0.588931im, atoms_gt[0] = 0.000000 + 0.588931im
atoms_epg_ssfp[1] = 0.000000 + 0.263514im, atoms_gt[1] = 0.000000 + 0.263514im
atoms_epg_ssfp[2] = 0.000000 + -0.056366im, atoms_gt[2] = 0.000000 + -0.056366im
atoms_epg_ssfp[3] = 0.000000 + -0.367502im, atoms_gt[3] = 0.000000 + -0.367502im
atoms_epg_ssfp[4] = 0.000000 + -0.667448im, atoms_gt[4] = 0.000000 + -0.667448im
The 2 th atom
atoms_epg_ssfp[0] = 0.000000 + 0.610385im, atoms_gt[0] = 0.000000 + 0.610385im
atoms_epg_ssfp[1] = 0.000000 + 0.526083im, atoms_gt[1] = 0.000000 + 0.526083im
atoms_epg_ssfp[2] = 0.000000 + 0.433221im, atoms_gt[2] = 0.000000 + 0.433221im
atoms_epg_ssfp[3] = 0.000000 + 0.333278im, atoms_gt[3] = 0.000000 + 0.333278im
atoms_epg_ssfp[4] = 0.000000 + 0.227843im, atoms_gt[4] = 0.000000 + 0.227843im
The 3 th atom
atoms_epg_ssfp[0] = 0.000000 + 0.588338im, atoms_gt[0] = 0.000000 + 0.588338im
atoms_epg_ssfp[1] = 0.000000 + 0.263248im, atoms_gt[1] = 0.000000 + 0.263248im
atoms_epg_ssfp[2] = 0.000000 + -0.056642im, atoms_gt[2] = 0.000000 + -0.056642im
atoms_epg_ssfp[3] = 0.000000 + -0.367906im, atoms_gt[3] = 0.000000 + -0.367906im
atoms_epg_ssfp[4] = 0.000000 + -0.667829im, atoms_gt[4] = 0.000000 + -0.667829im
The 4 th atom
atoms_epg_ssfp[0] = 0.000000 + 0.610948im, atoms_gt[0] = 0.000000 + 0.610948im
atoms_epg_ssfp[1] = 0.000000 + 0.526568im, atoms_gt[1] = 0.000000 + 0.526568im
atoms_epg_ssfp[2] = 0.000000 + 0.433275im, atoms_gt[2] = 0.000000 + 0.433275im
atoms_epg_ssfp[3] = 0.000000 + 0.332564im, atoms_gt[3] = 0.000000 + 0.332564im
atoms_epg_ssfp[4] = 0.000000 + 0.226148im, atoms_gt[4] = 0.000000 + 0.226148im
epg_ssfp unit test passed!

Reading ../data/atoms_mrf.ra
NREPS: 100, NATOMS: 109
Reading ../data/params_mrf.ra
Available GPU memory = 11301 MB
nsplits = 1, nminivoxels = 10
Matching...
The 1th voxel
T1_test[0] = 100.00, T1_gt[0] = 100.00
T2_test[0] = 70.00, T2_gt[0] = 70.00
B0_test[0] = 0.00, B0_gt[0] = 0.00
B1_test[0] = 1.00, B1_gt[0] = 1.00
The 2th voxel
T1_test[1] = 130.00, T1_gt[1] = 130.00
T2_test[1] = 20.00, T2_gt[1] = 20.00
B0_test[1] = 0.00, B0_gt[1] = 0.00
B1_test[1] = 1.00, B1_gt[1] = 1.00
The 3th voxel
T1_test[2] = 100.00, T1_gt[2] = 100.00
T2_test[2] = 20.00, T2_gt[2] = 20.00
B0_test[2] = 0.00, B0_gt[2] = 0.00
B1_test[2] = 1.00, B1_gt[2] = 1.00
The 4th voxel
T1_test[3] = 200.00, T1_gt[3] = 200.00
T2_test[3] = 90.00, T2_gt[3] = 90.00
B0_test[3] = 0.00, B0_gt[3] = 0.00
B1_test[3] = 1.00, B1_gt[3] = 1.00
The 5th voxel
T1_test[4] = 130.00, T1_gt[4] = 130.00
T2_test[4] = 100.00, T2_gt[4] = 100.00
B0_test[4] = 0.00, B0_gt[4] = 0.00
B1_test[4] = 1.00, B1_gt[4] = 1.00
The 6th voxel
T1_test[5] = 110.00, T1_gt[5] = 110.00
T2_test[5] = 90.00, T2_gt[5] = 90.00
B0_test[5] = 0.00, B0_gt[5] = 0.00
B1_test[5] = 1.00, B1_gt[5] = 1.00
The 7th voxel
T1_test[6] = 190.00, T1_gt[6] = 190.00
T2_test[6] = 10.00, T2_gt[6] = 10.00
B0_test[6] = 0.00, B0_gt[6] = 0.00
B1_test[6] = 1.00, B1_gt[6] = 1.00
The 8th voxel
T1_test[7] = 160.00, T1_gt[7] = 160.00
T2_test[7] = 40.00, T2_gt[7] = 40.00
B0_test[7] = 0.00, B0_gt[7] = 0.00
B1_test[7] = 1.00, B1_gt[7] = 1.00
The 9th voxel
T1_test[8] = 100.00, T1_gt[8] = 100.00
T2_test[8] = 90.00, T2_gt[8] = 90.00
B0_test[8] = 0.00, B0_gt[8] = 0.00
B1_test[8] = 1.00, B1_gt[8] = 1.00
The 10th voxel
T1_test[9] = 180.00, T1_gt[9] = 180.00
T2_test[9] = 50.00, T2_gt[9] = 50.00
B0_test[9] = 0.00, B0_gt[9] = 0.00
B1_test[9] = 1.00, B1_gt[9] = 1.00
Matching unit test passed!

Reading ../data/MRF_5.csv
Begin unit test for roa_ssfp
T1 = 100:100:200
T2 = 20+20:20:40+1000
B0 = 0
B1 = 1
l_t1: 2, l_t2: 3, l_b0: 1, l_b1: 1
nreps: 5, NATOMS: 4, natoms: 4
Constructing dictionary...
The 1 th atom
atoms_roa_ssfp[0] = 0.000000 + -0.484675im, atoms_gt[0] = 0.000000 + -0.484675im
atoms_roa_ssfp[1] = 0.000000 + -0.583299im, atoms_gt[1] = 0.000000 + -0.583298im
atoms_roa_ssfp[2] = 0.000000 + -0.509528im, atoms_gt[2] = 0.000000 + -0.509528im
atoms_roa_ssfp[3] = 0.000000 + -0.361696im, atoms_gt[3] = 0.000000 + -0.361696im
atoms_roa_ssfp[4] = 0.000000 + -0.185501im, atoms_gt[4] = 0.000000 + -0.185501im
The 2 th atom
atoms_roa_ssfp[0] = 0.000000 + -0.356310im, atoms_gt[0] = 0.000000 + -0.356310im
atoms_roa_ssfp[1] = 0.000000 + -0.483272im, atoms_gt[1] = 0.000000 + -0.483272im
atoms_roa_ssfp[2] = 0.000000 + -0.502212im, atoms_gt[2] = 0.000000 + -0.502212im
atoms_roa_ssfp[3] = 0.000000 + -0.468853im, atoms_gt[3] = 0.000000 + -0.468853im
atoms_roa_ssfp[4] = 0.000000 + -0.409208im, atoms_gt[4] = 0.000000 + -0.409208im
The 3 th atom
atoms_roa_ssfp[0] = 0.000000 + -0.366114im, atoms_gt[0] = 0.000000 + -0.366114im
atoms_roa_ssfp[1] = 0.000000 + -0.521149im, atoms_gt[1] = 0.000000 + -0.521149im
atoms_roa_ssfp[2] = 0.000000 + -0.534765im, atoms_gt[2] = 0.000000 + -0.534765im
atoms_roa_ssfp[3] = 0.000000 + -0.455552im, atoms_gt[3] = 0.000000 + -0.455552im
atoms_roa_ssfp[4] = 0.000000 + -0.317589im, atoms_gt[4] = 0.000000 + -0.317589im
The 4 th atom
atoms_roa_ssfp[0] = 0.000000 + -0.268246im, atoms_gt[0] = 0.000000 + -0.268246im
atoms_roa_ssfp[1] = 0.000000 + -0.422837im, atoms_gt[1] = 0.000000 + -0.422837im
atoms_roa_ssfp[2] = 0.000000 + -0.496872im, atoms_gt[2] = 0.000000 + -0.496872im
atoms_roa_ssfp[3] = 0.000000 + -0.513390im, atoms_gt[3] = 0.000000 + -0.513390im
atoms_roa_ssfp[4] = 0.000000 + -0.488674im, atoms_gt[4] = 0.000000 + -0.488674im
roa_ssfp unit test passed!

8 tests, 8 passed and 0 failed.
```

## Funding Sources
NIH R01 DK105371, NIH R01 EB017230, and NIH K25 CA176219.
