#ifndef _MRF_H
#define _MRF_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <err.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>
#include <sysexits.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "float2math.h"
#include "ra.h"

typedef struct ImagingParams
{
    float alpha;
    float phi;
    float TR;
    float TE;
} IP;

#ifdef __cplusplus
#endif

// Useful functions
void gpuAssert (cudaError_t code, const char *file, int line);
void * mallocAssert (size_t n, const char *file, int line);
void freeAssert (void *p, const char *file, int line);
void print_usage ();
size_t csv_dim (const char *filepath);
void csv_read (float *h_mrf, size_t nreps, const char *filepath);
void save_rafile (float2 * h_atoms, const char *atoms_path, size_t dim0,
    size_t dim1);
void save_rafile (float *h_params, const char *p_path, size_t dim0,
    size_t dim1);

// Parse params and compute natoms
void parse_range (const char *str, float *min, float *step, float *max,
    size_t * len);
size_t parse_length (const char *str);
void linspace (float *h_p, float min, float step, size_t len);
void parse_params (float *h_p, const char *str, size_t LEN);
void logspace(float *h_p, float min, float max, size_t len);
void trans_params (float *h_params, float *h_t1, float *h_t2, float *h_b0,
    float *h_b1, size_t l_t1, size_t l_t2, size_t l_b0, size_t l_b1,
    size_t NATOMS, int nparams);
size_t compute_natoms (float *h_t1, float *h_t2, size_t l_t1, size_t l_t2,
    size_t l_b0, size_t l_b1);

// Dictionary EPG
__device__ float parse_interval (float *d_mrf, size_t nreps, int index,
    int type);
__global__ void arrayscale (float *d_array, const size_t array_size,
    const float factor);
__global__ void arrayadd (float *d_array, const size_t array_size,
    const float factor);
__global__ void init_rf_pulse (float2 * d_w, float *d_params, size_t natoms,
    int nparams, int nstates, float alpha, float phi, float TI);
__global__ void fill_transition_matrix (float2 * d_T_m, float *d_mrf,
    size_t nreps, int index);
void apply_rf_pulse (float2 * d_w, float2 * d_T_m, size_t natoms,
    int nstates);
__global__ void compute_exp (float *d_exp, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int index, int type);
__global__ void decay_signal_stage1 (float2 * d_w, float *d_exp,
    size_t natoms, int nstates);
__global__ void decay_signal_stage2 (float2 * d_w, float *d_exp,
    size_t natoms, int nstates);
void decay_signal (float2 * d_w, float *d_params, float *d_mrf, float *d_exp,
    size_t nreps, size_t natoms, int nstates, int nparams, int index, int type);
__global__ void shift_phase (float2 * d_w, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int nstates, int index,
    int type);
__global__ void save_atom (float2 * d_atoms, float2 * d_w, size_t nreps,
    size_t NATOMS, int nstates, int idx_time, int idx_b1);
__global__ void dephase_gradients_stage1 (float2 * d_w, size_t natoms,
    int nstates);
__global__ void dephase_gradients_stage2 (float2 * d_w, size_t natoms,
    int nstates);
void dephase_gradients (float2 * d_w, size_t natoms, int nstates);
__global__ void normalize (float2 * d_atoms, size_t nreps, size_t NATOMS);
void epg_sr (float2 * d_atoms, float *d_params, float *d_mrf, size_t nreps,
    size_t natoms, int nparams, int nstates);
void epg_se (float2 * d_atoms, float *d_params, float *d_mrf, size_t nreps,
    size_t natoms, int nparams, int nstates);
void epg_tse (float2 * d_atoms, float *d_params, float *d_mrf, size_t nreps,
    size_t natoms, int nparams, int nstates);
void epg_fisp (float2 * d_atoms, float *d_params, float *d_mrf, size_t nreps,
    size_t natoms, int nparams, int nstates);
void epg_ssfp (float2 * d_atoms, float *d_params, float *d_mrf, float TI,
    size_t nreps, size_t NATOMS, size_t natoms, int nparams, int nstates);

// Dictionary ROA
__global__ void init_rf_pulse (float * d_w, float *d_params, size_t natoms,
    int nparams, float alpha, float phi, float TI);
__global__ void fill_transition_matrix (float *d_T_m, float *d_mrf,
  size_t nreps, int index);
void apply_rf_pulse (float *d_w, float *d_T_m, size_t natoms);
__global__ void decay_signal (float *d_w, float *d_exp, size_t natoms);
__global__ void fill_offresoance_operator (float *d_rbeta, float *d_mrf,
    float *d_params, size_t natoms, size_t nreps, int nparams, int index);
__global__ void shift_phase (float *d_w, float *d_rbeta, size_t natoms);
__global__ void save_atom (float2 * d_atoms, float *d_w, size_t nreps,
    size_t natoms, int index);
void roa_ssfp (float2 * d_atoms, float *d_mrf, float *d_params, size_t nreps,
    size_t natoms, int nparams);
void MRF_dict (float2 * d_atoms, float *d_params, float *h_mrf,
    size_t nreps, size_t NREPS, size_t NATOMS, size_t natoms,
    int nparams, int nstates, const char *echo_type);


// Matching
__global__ void generate_maps (float *d_map, float2 * d_mat, float *d_params,
    size_t NATOMS, size_t nvoxels, int nparams, int nmaps);
__global__ void merge_maps (float *d_maps, float *d_MAPS, float *d_b1map,
    size_t NATOMS, size_t natoms, size_t nvoxels, int nmaps);
__global__ void merge_maps (float *d_maps, float *d_MAPS,
    size_t NATOMS, size_t natoms, size_t nvoxels, int nmaps);
void compute_nsplits (size_t * nsplits, size_t * nminivoxels, size_t nvoxels,
    size_t NATOMS, int type);
void MRF_minimatch (float *d_maps, float2 * d_img, float2 * d_atoms,
    float2 * d_mat, float *d_params, size_t nreps, size_t NATOMS,
    size_t mininvoxels, int nparams, int nmaps);
void MRF_match(float *d_maps, float2 *h_img, float2 *d_atoms, float *d_params,
    size_t NREPS, size_t nreps, size_t NATOMS, size_t natoms, size_t nvoxels,
    int nparams, int nmaps, const char *in_b1map);

// Unittest
void generate_echo (float *h_echo, size_t nreps, IP ip);
void compare (int *pass, int *fail, float2 * d_atoms, size_t nreps,
    size_t NATOMS, const char *echo_type);
void unittest_dictionary (int *pass, int *fail, float *h_echo, size_t nreps,
    size_t NREPS, int nparams, int nstates, const char *T1, const char *T2,
    const char *B0, const char *B1, const char *echo_type);
void unittest_matching (int *pass, int *fail, const char *atoms_path,
    const char *params_path, int nparams, int nmaps, int ntests);

#ifdef __cplusplus
#endif

#ifndef MACROS_H_
#define MACROS_H_

#define cuTry(ans) gpuAssert((ans), __FILE__, __LINE__);
#define safe_malloc(n) mallocAssert(n, __FILE__, __LINE__);
#define safe_free(x) freeAssert(x, __FILE__, __LINE__);
#define min(a,b) ((a<b) ? a : b);

#endif

#endif
