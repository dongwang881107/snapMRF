/*
  This file is part of the MRF_CUDA package (https://github.com/chixindebaoyu/MRF_CUDA)

  The MIT License (MIT)

  Copyright (c) 2019 Dong Wang and David Smith

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  coM_PIes of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  coM_PIes or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include "functions.h"

// Specify the sizes according to the divice
int blocksize = 64;
int gridsize = 128;

cudaStream_t stream[2];
cublasHandle_t handle;
cublasStatus_t stat;

//// Useful functions
void
gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
        getchar();
        exit(code);
    }
}

#define cuTry(ans) gpuAssert((ans), __FILE__, __LINE__);

void *
mallocAssert(size_t n, const char *file, int line)
{
    void *p = malloc(n);
    if (p == NULL)
    {
        printf("Malloc failed from %s:%d", file, line);
        abort();
    }
    return p;
}

#define safe_malloc(n) mallocAssert(n, __FILE__, __LINE__);

void
freeAssert(void *p, const char *file, int line)
{
    if (p == NULL)
    {
        printf("Free failed from %s:%d", file, line);
        abort();
    }
}

#define safe_free(x) freeAssert(x, __FILE__, __LINE__);

void
print_usage()
{
	fprintf(stderr, "MR fingerprinting dictionary generator and image reconstructor\n");
    fprintf(stderr, "Usage: mrf [OPTION] <rfpulse.csv> <imgstack.ra>\n");
    fprintf(stderr, "\t-M, --b1map <rafile>\t\t B1 map input RA file\n");
    fprintf(stderr, "\t-N, --NATOMS\t\t\t number of atoms\n");
    fprintf(stderr, "\t-a, --atoms <rafile>\t\t dictionary atoms output RA file\n");
    fprintf(stderr, "\t-m, --maps  <rafile>\t\t parameter map output RA file\n");
    fprintf(stderr, "\t-p, --params <rafile>\t\t dictionary parameters\n");
    fprintf(stderr, "\t-t, --T1 start:step:end\t\t T1 grid\n");
    fprintf(stderr, "\t-s, --T2 start:step:end\t\t T2 grid\n");
    fprintf(stderr, "\t-b, --B0 start:step:end\t\t B0 grid\n");
    fprintf(stderr, "\t-r, --B1 start:step:end\t\t B1 grid\n");
    fprintf(stderr, "\t-w, --states n\t\t\t number of historical states to track\n");
    fprintf(stderr, "\t-e, --echo_type n\t\t echo type\n");
    fprintf(stderr, "\t-T, --timepoints n\t\t number of time points\n");
    fprintf(stderr, "\t-G, --gridsize n\t\t set GPU gridsize\n");
    fprintf(stderr, "\t-B, --blocksize n\t\t set GPU blocksize\n");
    fprintf(stderr, "\t-h\t\t\t\t show this help\n");
}

size_t
csv_dim(const char *filepath)
{
    FILE *file = fopen(filepath, "r");

    if (file == NULL)
    {
        printf("Unable to open file %s\n", filepath);
        exit(1);
    }

    size_t row = 0;
    for (char c = getc(file); c != EOF; c = getc(file))
    {
        if (c == '\n')
            row = row + 1;
    }

    fclose(file);
    return row;
}

void
csv_read(float *h_mrf, size_t nreps, const char *filepath)
{
    // read csv file and save in h_mrf
    // assume the file contains 4 columns
    FILE *file = fopen(filepath, "r");
    char buffer[100];
    char *tmp;

    if (file == NULL)
    {
        printf("Unable to open file %s\n", filepath);
        exit(1);
    }

    int index = 0;
    int count = 0;
    while (fgets(buffer, 100, file) && (index < nreps))
    {
        if (count > 1)
        {
            tmp = strtok(buffer, ",");
            h_mrf[0 * nreps + index] = atof(tmp);
            tmp = strtok(NULL, ",");
            h_mrf[1 * nreps + index] = atof(tmp);
            tmp = strtok(NULL, ",");
            h_mrf[2 * nreps + index] = atof(tmp);
            tmp = strtok(NULL, ",");
            h_mrf[3 * nreps + index] = atof(tmp);

            index++;
        }
        count++;
    }

    fclose(file);
}

void
save_rafile(float2 * h_atoms, const char *atoms_path, size_t dim0, size_t dim1)
{
    // save atoms
    ra_t atoms;
    atoms.flags = 0;
    atoms.eltype = RA_TYPE_COMPLEX;
    atoms.elbyte = sizeof(float2);
    atoms.size = dim0 * dim1 * sizeof(float2);
    atoms.ndims = 2;
    atoms.dims = (uint64_t *) safe_malloc(atoms.ndims * sizeof(uint64_t));
    atoms.dims[0] = dim0;
    atoms.dims[1] = dim1;
    atoms.data = (uint8_t *) h_atoms;
    ra_write(&atoms, atoms_path);
    ra_free(&atoms);
}

void
save_rafile(float *h_params, const char *p_path, size_t dim0, size_t dim1)
{
    // save params
    ra_t p;
    p.flags = 0;
    p.eltype = RA_TYPE_FLOAT;
    p.elbyte = sizeof(float);
    p.size = dim0 * dim1 * sizeof(float);
    p.ndims = 2;
    p.dims = (uint64_t *) safe_malloc(p.ndims * sizeof(uint64_t));
    p.dims[0] = dim0;
    p.dims[1] = dim1;
    p.data = (uint8_t *) h_params;
    ra_write(&p, p_path);
    ra_free(&p);
}

//// Parse params and compute natoms
void
parse_range(const char *str, float *min, float *step, float *max,
    size_t * len)
{
    float tmp;

    tmp = atof(str);
    *min = tmp;
    str = strchr(str, ':');
    if (str == NULL)
    {
        *step = 0;
        *max = tmp;

        *len = 1;
    }
    else
    {
        str = str + 1;
        tmp = atof(str);
        *step = tmp;
        str = strchr(str, ':');
        if (str == NULL)
        {
            *step = 1;
            *max = tmp;
            assert(*max >= *min);
            *len = (size_t) (((*max - *min) / 1) + 1);
        }
        else
        {
            str = str + 1;
            tmp = atof(str);
            *max = tmp;
            assert(*max >= *min);
            *len = (size_t) (((*max - *min) / *step) + 1);
        }
    }
}

size_t
parse_length(const char *str)
{
    char *range;
    float min, step, max, tmp;
    size_t len;

    char str_cpy[1024];
    strcpy(str_cpy, str);

    int LEN = 0;
    int count = 0;
    range = strtok(str_cpy, "+");
    while (range != NULL)
    {
        count++;
        parse_range(range, &min, &step, &max, &len);
        if (count == 1)
            tmp = min + (len - 1) * step;
        else
        {
            if (abs(tmp - min) < 1E-7)
                len = len - 1;
            tmp = min + len * step;
        }
        LEN += len;
        range = strtok(NULL, "+");
    }

    return LEN;
}

void
linspace(float *h_p, float min, float step, size_t len)
{
    for (int i = 0; i < len; i++)
        h_p[i] = min + i * step;
}

void
logspace(float *h_p, float min, float max, size_t len)
{
    float tmp;
    if (min > max)
    {
        tmp = min;
        min = max;
        max = tmp;
    }

    float logmin, logstep, logmax;
    if (len == 1)
        h_p[0] = (min + max) / 2.f;
    else
    {
        if (min > 0.f)
        {
            logmin = log10(min);
            logmax = log10(max);
            logstep = (logmax - logmin) / (len - 1);

            for (int i = 0; i < len; i++)
                h_p[i] = pow(10.f, logmin + i * logstep);
        }
        else
        {
            logmax = log10(max);
            logstep = logmax / (len / 2);

            for (int i = 0; i < len / 2 + 1; i++)
                h_p[i] = -pow(10.f, (len / 2 - i) * logstep);
            h_p[len / 2] = 0.f;
            for (int i = 0; i < len / 2 + 1; i++)
                h_p[i + len / 2 + 1] = pow(10.f, (i + 1) * logstep);
        }
    }
}

void
parse_params(float *h_p, const char *str, size_t LEN)
{
    char *range;
    float min, step, max, tmp;
    size_t len;

    char str_cpy[1024];
    strcpy(str_cpy, str);

    int stride = 0;
    int count = 0;
    range = strtok(str_cpy, "+");
    while (range != NULL)
    {
        count++;
        parse_range(range, &min, &step, &max, &len);
        if (count == 1)
        {
            tmp = min + (len - 1) * step;
            linspace(h_p + stride, min, step, len);
        }
        else
        {
            if (abs(tmp - min) < 1E-7)
            {
                len = len - 1;
                linspace(h_p + stride, min + step, step, len);
            }
            else
                linspace(h_p + stride, min, step, len);
            tmp = min + len * step;
        }
        stride += len;
        range = strtok(NULL, "+");
    }

    assert(stride == LEN);
}

void
trans_params(float *h_params,
    float *h_t1, float *h_t2, float *h_b0, float *h_b1,
    size_t l_t1, size_t l_t2, size_t l_b0, size_t l_b1, size_t NATOMS,
    int nparams)
{
    float T1, T2, B0, B1;
    int count = -1;

    for (int h = 0; h < l_b1; h++)
    {
        B1 = h_b1[h];
        for (int i = 0; i < l_b0; i++)
        {
            B0 = h_b0[i];
            for (int j = 0; j < l_t2; j++)
            {
                T2 = h_t2[j];
                for (int k = 0; k < l_t1; k++)
                {
                    T1 = h_t1[k];
                    if (T1 > T2)
                    {
                        count++;
                        h_params[count * nparams] = T1;
                        h_params[count * nparams + 1] = T2;
                        h_params[count * nparams + 2] = B0;
                        h_params[count * nparams + 3] = B1;
                    }
                }
            }
        }
    }
    count++;

    assert(count == NATOMS);
}

size_t
compute_natoms(float *h_t1, float *h_t2, size_t l_t1, size_t l_t2,
    size_t l_b0, size_t l_b1)
{
    size_t NATOMS = 0;
    float T1, T2;

    for (int i = 0; i < l_t1; i++)
    {
        T1 = h_t1[i];

        for (int j = 0; j < l_t2; j++)
        {
            T2 = h_t2[j];

            if (T1 > T2)
                NATOMS++;
        }
    }

    return NATOMS * l_b0 * l_b1;
}

// EPG Dictionary
__device__ float
parse_interval(float *d_mrf, size_t nreps, int index, int type)
{
    float TE, TR;

    TE = d_mrf[3 * nreps + index];
    TR = d_mrf[2 * nreps + index];

	assert(1 <= type);
	assert(type <= 4);

    if (type == 1)
        return TE;
    else if (type == 2)
        return TR - TE;
    else if (type == 3)
        return TE/2;
    else if (type == 4)
        return TR;
    else
		    return 0.f;  // never get here, hopefully
}

__global__ void
arrayscale(float *d_array, const size_t array_size, const float factor)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
        id += blockDim.x * gridDim.x)
        d_array[id] *= factor;
}

__global__ void
arrayadd(float *d_array, const size_t array_size, const float factor)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < array_size;
        id += blockDim.x * gridDim.x)
        d_array[id] += factor;
}

__global__ void
init_rf_pulse(float2 * d_w, float *d_params, size_t natoms, int nparams,
    int nstates, float alpha, float phi, float TI = 0.f)
{
    float t1, t2, e1, e2, sa, ca, sph, cph;
    float2 tmp2;

    alpha = alpha * M_PI / 180.f;
    phi = phi * M_PI / 180.f;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        t1 = d_params[id * nparams];
        t2 = d_params[id * nparams + 1];
        e1 = expf(-TI / t1);
        e2 = expf(-TI / t2);
        __sincosf(alpha, &sa, &ca);
        __sincosf(phi, &sph, &cph);
        tmp2 = e2 * sa * make_float2(sph,cph);

        d_w[id * 3 * nstates] = conj(tmp2);
        d_w[1 + id * 3 * nstates] = tmp2;
        d_w[2 + id * 3 * nstates] = make_float2(e1 * ca + 1 - e1);
    }
}

__global__ void
fill_transition_matrix(float2 * d_T_m, float *d_mrf, size_t nreps, int index)
{
    float sa, ca, sa2, ca2, alpha, phi;
    float2 z, zinv;
    const float2 im = make_float2(0.f, 1.f);

    alpha = d_mrf[index];
    phi = d_mrf[nreps + index];

    __sincosf(alpha, &sa, &ca);
    sa2 = (1 - ca) / 2;
    ca2 = (1 + ca) / 2;
    z = cexpf(phi);
    zinv = conj(z);

    d_T_m[0] = make_float2(ca2);
    d_T_m[3] = sa2 * zinv * zinv;
    d_T_m[6] = -im * 0.5f * sa * zinv;
    d_T_m[1] = z * z * sa2;
    d_T_m[4] = make_float2(ca2);
    d_T_m[7] = im * 0.5f * z * sa;
    d_T_m[2] = -im * z * sa;
    d_T_m[5] = im * sa * zinv;
    d_T_m[8] = make_float2(ca);
}

void
apply_rf_pulse(float2 * d_w, float2 * d_T_m, size_t natoms, int nstates)
{
    float2 alpha = make_float2(1.f);
    float2 beta = make_float2(0.f);

    stat = cublasCreate(&handle);

    cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3, nstates * natoms, 3,
        &alpha, d_T_m, 3, d_w, 3, &beta, d_w, 3);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS Cgemm failed:%d\n", stat);
        exit(EXIT_FAILURE);
    }

    cublasDestroy(handle);
}

__global__ void
compute_exp(float *d_exp, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int index, int type)
{
    float t = parse_interval(d_mrf, nreps, index, type);

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        d_exp[2 * id] = expf(-t / d_params[id * nparams]);
        d_exp[2 * id + 1] = expf(-t / d_params[id * nparams + 1]);
    }
}

__global__ void
decay_signal_stage1(float2 * d_w, float *d_exp, size_t natoms, int nstates)
{
    int idx;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < nstates * natoms; id += blockDim.x * gridDim.x)
    {
        idx = id / nstates * 2;

        d_w[3 * id] = d_exp[idx + 1] * d_w[3 * id];
        d_w[3 * id + 1] = d_exp[idx + 1] * d_w[3 * id + 1];
        d_w[3 * id + 2] = d_exp[idx] * d_w[3 * id + 2];
    }
}

__global__ void
decay_signal_stage2(float2 * d_w, float *d_exp, size_t natoms, int nstates)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        d_w[3 * nstates * id + 2] += make_float2(1 - d_exp[2 * id]);
    }
}

void
decay_signal(float2 * d_w, float *d_params, float *d_mrf, float *d_exp,
    size_t nreps, size_t natoms, int nstates, int nparams, int index, int type)
{
    compute_exp<<<gridsize, blocksize>>>(d_exp, d_params, d_mrf, nreps,
        natoms, nparams, index, type);
    cudaDeviceSynchronize();
    decay_signal_stage1<<<gridsize, blocksize>>>(d_w, d_exp, natoms,
        nstates);
    cudaDeviceSynchronize();
    decay_signal_stage2<<<gridsize, blocksize>>>(d_w, d_exp, natoms,
        nstates);
    cudaDeviceSynchronize();
}

__global__ void
shift_phase(float2 * d_w, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int nstates, int index,
    int type)
{
    // shift phase for B0
    float t = parse_interval(d_mrf, nreps, index, type);

    float B0;
    float2 e;
    int idx;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < nstates * natoms; id += blockDim.x * gridDim.x)
    {
        idx = id / nstates * nparams + 2;
        B0 = d_params[idx];
        e = cexpf(2 * M_PI * B0 * t / 1000.f);

        d_w[3 * id] = e * d_w[3 * id];
        d_w[3 * id + 1] = conj(e) * d_w[3 * id + 1];
    }
}

__global__ void
save_atom(float2 * d_atoms, float2 * d_w, size_t nreps, size_t natoms,
    int nstates, int idx_time, int idx_b1)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
        d_atoms[idx_b1 * nreps * natoms + id * nreps + idx_time] =
            d_w[id * 3 * nstates];
}

__global__ void
dephase_gradients_stage1(float2 * d_w, size_t natoms, int nstates)
{
    int j, idx, stride;
    int from_idx, to_idx;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < 2 * natoms;
        id += blockDim.x * gridDim.x)
    {
        idx = id % 2;
        stride = (id / 2) * 3 * nstates;

        for (int i = 0; i < nstates - 1; i++)
        {
            j = nstates - 1 - i;

            from_idx = (1 - idx) * j + idx * i;
            to_idx = from_idx + 2 * idx - 1;

            d_w[stride + 3 * from_idx + idx] =
                d_w[stride + 3 * to_idx + idx];
        }
    }
}

__global__ void
dephase_gradients_stage2(float2 * d_w, size_t natoms, int nstates)
{
    int idx;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        idx = id * 3 * nstates;

        d_w[idx] = conj(d_w[idx + 1]);
        d_w[idx + 3 * nstates - 2] = make_float2(0.f);
    }
}

void
dephase_gradients(float2 * d_w, size_t natoms, int nstates)
{
    dephase_gradients_stage1<<<gridsize, blocksize, 0,
        stream[1]>>>(d_w, natoms, nstates);
    cudaDeviceSynchronize();
    dephase_gradients_stage2<<<gridsize, blocksize, 0,
        stream[1]>>>(d_w, natoms, nstates);
    cudaDeviceSynchronize();
}

__global__ void
normalize(float2 * d_atoms, size_t nreps, size_t NATOMS)
{
    float array_norm;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < NATOMS;
        id += blockDim.x * gridDim.x)
    {
        array_norm = 0.f;
        for (int i = id * nreps; i < (id * nreps + nreps); i++)
            array_norm += norm(d_atoms[i]);
        if (array_norm > 0.f)
            array_norm = 1.f / sqrtf(array_norm);
        for (int i = id * nreps; i < (id * nreps + nreps); i++)
            d_atoms[i] = d_atoms[i] * array_norm;
    }
}

void
epg_sr(float2 * d_atoms, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int nstates)
{
    // Simulating Saturation-Recovery sequence
    // RF pulses are nreps 60-degree exciation pulses about y axis (phi is 90-degree)
    // Long TR and short TE gurantee that transverse magnetization dies out before
    // the next excitation, so no dephase gradients

    // transition matrix
    float2 *d_T_m;
    cuTry(cudaMalloc((void **) &d_T_m, 9 * sizeof(float2)));

    float2 *d_w;
    cuTry(cudaMalloc((void **) &d_w,
            3 * nstates * natoms * sizeof(float2)));
    cuTry(cudaMemsetAsync(d_w, 0, 3 * nstates * natoms * sizeof(float2),
            stream[0]));

    float *d_exp;
    cuTry(cudaMalloc((void **) &d_exp, 2 * natoms * sizeof(float)));

    init_rf_pulse<<<gridsize, blocksize, 0, stream[0]>>>(d_w, d_params,
        natoms, nparams, nstates, 0.f, 0.f);
    cudaDeviceSynchronize();

    for (int i = 0; i < nreps; i++)
    {
        // flip angle transition
        fill_transition_matrix<<<gridsize, blocksize, 0,
            stream[1]>>>(d_T_m, d_mrf, nreps, i);
        cudaDeviceSynchronize();

        apply_rf_pulse(d_w, d_T_m, natoms, nstates);

        // decay at TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 1);

        // store readout transverse magnetization
        save_atom<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms, d_w,
            nreps, natoms, nstates, i, 0);
        cudaDeviceSynchronize();

        // decay for flip angle at TR-TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 2);
    }

    cuTry(cudaFree(d_w));
    cuTry(cudaFree(d_T_m));
    cuTry(cudaFree(d_exp));
}

void
epg_se(float2 * d_atoms, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int nstates)
{
    // Simulating (Fast) SM_PIn Echo sequence
    // RF pulses are one 90-degree about y and
    // nreps 180-degree about x

    // transition matrix
    float2 *d_T_m;
    cuTry(cudaMalloc((void **) &d_T_m, 9 * sizeof(float2)));

    // configuration states matrices
    float2 *d_w;
    cuTry(cudaMalloc((void **) &d_w,
            3 * nstates * natoms * sizeof(float2)));
    cuTry(cudaMemsetAsync(d_w, 0, 3 * nstates * natoms * sizeof(float2),
            stream[0]));

    float *d_exp;
    cuTry(cudaMalloc((void **) &d_exp, 2 * natoms * sizeof(float)));

    // initialize
    init_rf_pulse<<<gridsize, blocksize, 0, stream[0]>>>(d_w, d_params,
        natoms, nparams, nstates, 90.f, 90.f);
    cudaDeviceSynchronize();

    // gradient dephasing
    dephase_gradients(d_w, natoms, nstates);

    // dacay at TE
    decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates, nparams,
      0, 1);

    for (int i = 0; i < nreps; i++)
    {
        // flip angle transition
        fill_transition_matrix<<<gridsize, blocksize, 0,
            stream[1]>>>(d_T_m, d_mrf, nreps, i);
        cudaDeviceSynchronize();

        apply_rf_pulse(d_w, d_T_m, natoms, nstates);

        // gradient dephasing
        dephase_gradients(d_w, natoms, nstates);

        // decay for readout at TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 1);

        // store readout transverse magnetization
        save_atom<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms, d_w,
            nreps, natoms, nstates, i, 0);
        cudaDeviceSynchronize();

        // gradient dephasing
        dephase_gradients(d_w, natoms, nstates);

        // decay for flip angle at TR-TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 2);
    }

    cuTry(cudaFree(d_w));
    cuTry(cudaFree(d_T_m));
    cuTry(cudaFree(d_exp));
}

void
epg_tse(float2 * d_atoms, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int nstates)
{
    // Stimulating Turbo SM_PIn Echo sequence neglecting relaxation effects
    // Assume TR is long enough
    // Extended phase graphs: dephasing, RF pulses, and echoes - pure and simple
    // RF pulses are 90-120-120-120

    // transition matrix
    float2 *d_T_m;
    cuTry(cudaMalloc((void **) &d_T_m, 9 * sizeof(float2)));

    // configuration states matrices
    float2 *d_w;
    cuTry(cudaMalloc((void **) &d_w,
            3 * nstates * natoms * sizeof(float2)));
    cuTry(cudaMemsetAsync(d_w, 0, 3 * nstates * natoms * sizeof(float2),
            stream[0]));

    float *d_exp;
    cuTry(cudaMalloc((void **) &d_exp, 2 * natoms * sizeof(float)));

    // initialize with 90 degree RF pulse
    init_rf_pulse<<<gridsize, blocksize, 0, stream[0]>>>(d_w, d_params,
        natoms, nparams, nstates, 90.f, 90.f);
    cudaDeviceSynchronize();

    // gradient dephasing
    dephase_gradients(d_w, natoms, nstates);

    // decay at TE
    decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates, nparams,
       0, 1);

    for (int i = 0; i < nreps; i++)
    {
        // flip angle transition
        fill_transition_matrix<<<gridsize, blocksize, 0,
            stream[1]>>>(d_T_m, d_mrf, nreps, i);
        cudaDeviceSynchronize();

        apply_rf_pulse(d_w, d_T_m, natoms, nstates);

        // gradient dephasing
        dephase_gradients(d_w, natoms, nstates);

        // decay at TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 1);

        // store readout transverse magnetization
        save_atom<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms, d_w,
            nreps, natoms, nstates, i, 0);
        cudaDeviceSynchronize();

        // gradient dephasing
        dephase_gradients(d_w, natoms, nstates);

        // decay at TR-TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 2);
    }

    cuTry(cudaFree(d_w));
    cuTry(cudaFree(d_T_m));
    cuTry(cudaFree(d_exp));
}

void
epg_fisp(float2 * d_atoms, float *d_params, float *d_mrf,
    size_t nreps, size_t natoms, int nparams, int nstates)
{
    // Stimulating a F0 type SSFP (gradient spoiling but no RF spoiling)
    // Extended phase graphs: dephasing, RF pulses, and echoes - pure and simple

    // transition matrix
    float2 *d_T_m;
    cuTry(cudaMalloc((void **) &d_T_m, 9 * sizeof(float2)));

    // configuration states matrices
    float2 *d_w;
    cuTry(cudaMalloc((void **) &d_w,
            3 * nstates * natoms * sizeof(float2)));
    cuTry(cudaMemsetAsync(d_w, 0, 3 * nstates * natoms * sizeof(float2),
            stream[0]));

    float *d_exp;
    cuTry(cudaMalloc((void **) &d_exp, 2 * natoms * sizeof(float)));

    // initialize
    init_rf_pulse<<<gridsize, blocksize, 0, stream[0]>>>(d_w, d_params,
        natoms, nparams, nstates, 0.f, 0.f);
    cudaDeviceSynchronize();

    for (int i = 0; i < nreps; i++)
    {
        // flip angle transition
        fill_transition_matrix<<<gridsize, blocksize, 0,
            stream[1]>>>(d_T_m, d_mrf, nreps, i);
        cudaDeviceSynchronize();

        apply_rf_pulse(d_w, d_T_m, natoms, nstates);

        // decay at TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 1);

        // store readout transverse magnetization
        save_atom<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms, d_w,
            nreps, natoms, nstates, i, 0);
        cudaDeviceSynchronize();

        // gradient dephasing
        dephase_gradients(d_w, natoms, nstates);

        // decay at TR-TE
        decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
            nparams, i, 2);
    }

    cuTry(cudaFree(d_w));
    cuTry(cudaFree(d_T_m));
    cuTry(cudaFree(d_exp));
}

void
epg_ssfp(float2 * d_atoms, float *d_params, float *d_mrf,
    float TI, size_t nreps, size_t NATOMS, size_t natoms,
    int nparams, int nstates)
{
    // Stimulating mrf SSFP using extended phase graph

    // transition matrix
    float2 *d_T_m;
    cuTry(cudaMalloc((void **) &d_T_m, 9 * sizeof(float2)));

    float2 *d_w;
    cuTry(cudaMalloc((void **) &d_w,
            3 * nstates * natoms * sizeof(float2)));
    cuTry(cudaMemsetAsync(d_w, 0, 3 * nstates * natoms * sizeof(float2),
            stream[0]));

    float *d_exp;
    cuTry(cudaMalloc((void **) &d_exp, 2 * natoms * sizeof(float)));

    float b1;

    // begin iteration
    for (int j = 0; j < NATOMS/natoms; j++)
    {
        // reset d_w to 0
        cuTry(cudaMemsetAsync(d_w, 0.f, 3 * nstates * natoms * sizeof(float2),
              stream[0]));

        // initialize
        init_rf_pulse<<<gridsize, blocksize, 0, stream[0]>>>(d_w, d_params,
            natoms, nparams, nstates, 180.f, 0.f, TI);
        cudaDeviceSynchronize();

        // multiply flip angle by B1
        cuTry(cudaMemcpy(&b1, d_params+j*natoms*nparams+3,
            sizeof(float), cudaMemcpyDeviceToHost));
        arrayscale<<<gridsize, blocksize, 0, stream[0]>>>
            (d_mrf, nreps, b1);

        for (int i = 0; i < nreps; i++)
        {
            // flip angle transition
            fill_transition_matrix<<<gridsize, blocksize, 0,
                stream[1]>>>(d_T_m, d_mrf, nreps, i);
            cudaDeviceSynchronize();

            apply_rf_pulse(d_w, d_T_m, natoms, nstates);

            // decay for readout at TE
            decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
                nparams, i, 1);

            // shift phase at TE
            shift_phase<<<gridsize, blocksize, 0, stream[0]>>>(d_w,
                d_params, d_mrf, nreps, natoms, nparams, nstates, i, 1);
            cudaDeviceSynchronize();

            // store readout transverse magnetization
            save_atom<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms, d_w,
                nreps, natoms, nstates, i, j);
            cudaDeviceSynchronize();

            // shift phase at TR-TE
            shift_phase<<<gridsize, blocksize, 0, stream[0]>>>(d_w,
                d_params, d_mrf, nreps, natoms, nparams, nstates, i, 2);
            cudaDeviceSynchronize();

            // Gradient dephasing specified by delk = 1
            dephase_gradients(d_w, natoms, nstates);

            // decay for flip angle at TR-TE
            decay_signal(d_w, d_params, d_mrf, d_exp, nreps, natoms, nstates,
                nparams, i, 2);
        }

        // reset d_w and d_mrf
        arrayscale<<<gridsize, blocksize, 0, stream[0]>>>
            (d_mrf, nreps, 1.f/b1);
    }

    cuTry(cudaFree(d_w));
    cuTry(cudaFree(d_T_m));
    cuTry(cudaFree(d_exp));
}

// ROA Dictionary
__global__ void
init_rf_pulse(float *d_w, float *d_params, size_t natoms, int nparams,
    float alpha, float phi, float TI = 0.f)
{
    float t1, t2, e1, e2, sa, ca, sph, cph, sph2, cph2;

    alpha = alpha * M_PI / 180.f;
    phi = phi * M_PI / 180.f;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
      t1 = d_params[id * nparams];
      t2 = d_params[id * nparams + 1];
      e1 = expf(-TI / t1);
      e2 = expf(-TI / t2);
      __sincosf(alpha, &sa, &ca);
      __sincosf(phi, &sph, &cph);
      sph2 = 2 * cph * sph;
      cph2 = cph * cph - sph * sph;

      d_w[id * 3] = e2 * sph2 * sa;
      d_w[1 + id * 3] = e2 * cph2 * sa;
      d_w[2 + id * 3] = e1 * ca + 1 - e1;
    }
}

__global__ void
fill_transition_matrix(float *d_T_m, float *d_mrf, size_t nreps, int index)
{
    float alpha, phi;

    alpha = d_mrf[index];
    phi = d_mrf[nreps + index];

    // multiply the three matrices together
    float sa, ca, sph, cph, sph2, cph2;

    __sincosf(alpha, &sa, &ca);
    __sincosf(phi, &sph, &cph);

    sph2 = 2 * cph * sph;
    cph2 = cph * cph - sph * sph;

    d_T_m[0] = cph2;
    d_T_m[1] = -sph2;
    d_T_m[2] = 0.f;
    d_T_m[3] = sph2 * ca;
    d_T_m[4] = cph2 * ca;
    d_T_m[5] = -sa;
    d_T_m[6] = sph2 * sa;
    d_T_m[7] = cph2 * sa;
    d_T_m[8] = ca;
}

void
apply_rf_pulse(float *d_w, float *d_T_m, size_t natoms)
{
    float alpha = 1.f;
    float beta = 0.f;

    stat = cublasCreate(&handle);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, natoms, 3, &alpha,
        d_T_m, 3, d_w, 3, &beta, d_w, 3);

    if (stat != CUBLAS_STATUS_SUCCESS)
        err(EX_SOFTWARE, "CUBLAS Sgemm failed:%d\n", stat);

    cublasDestroy(handle);
}

__global__ void
decay_signal(float *d_w, float *d_exp, size_t natoms)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        d_w[3 * id] = d_exp[2 * id + 1] * d_w[3 * id];
        d_w[3 * id + 1] = d_exp[2 * id + 1] * d_w[3 * id + 1];
        d_w[3 * id + 2] =
            d_exp[2 * id] * d_w[3 * id + 2] + (1 - d_exp[2 * id]);
    }
}

__global__ void
fill_offresoance_operator(float *d_rbeta, float *d_mrf, float *d_params,
    size_t natoms, size_t nreps, int nparams, int index)
{
    float beta, B0, TR;
    float sb, cb;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        B0 = d_params[id * nparams + 2];
        TR = d_mrf[2 * nreps + index];

        beta = M_PI * B0 * TR / 1000.f;
        __sincosf(beta, &sb, &cb);

        d_rbeta[2 * id] = cb;
        d_rbeta[2 * id + 1] = sb;
    }
}

__global__ void
shift_phase(float *d_w, float *d_rbeta, size_t natoms)
{
    float w1, w2, cb, sb;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
    {
        w1 = d_w[3 * id];
        w2 = d_w[3 * id + 1];
        cb = d_rbeta[2 * id];
        sb = d_rbeta[2 * id + 1];

        d_w[3 * id] = cb * w1 + sb * w2;
        d_w[3 * id + 1] = cb * w2 - sb * w1;
    }
}

__global__ void
save_atom(float2 * d_atoms, float *d_w, size_t nreps, size_t natoms,
    int index)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < natoms;
        id += blockDim.x * gridDim.x)
        d_atoms[id * nreps + index] =
            make_float2(d_w[3 * id], d_w[3 * id + 1]);
}

void
roa_ssfp(float2 * d_atoms, float *d_mrf, float *d_params,
    size_t nreps, size_t natoms, int nparams)
{
    // Stimulating mrf SSFP using Rotation Operators

    // Rotation operator
    float *d_T_m;
    cuTry(cudaMalloc((void **) &d_T_m, 9 * sizeof(float)));

    float *d_rbeta;
    cuTry(cudaMalloc((void **) &d_rbeta, 2 * natoms * sizeof(float)));

    // State matrix
    float *d_w;
    cuTry(cudaMalloc((void **) &d_w, 3 * natoms * sizeof(float)));
    cuTry(cudaMemsetAsync(d_w, 0.f, 3 * natoms * sizeof(float),
            stream[0]));

    float *d_exp;
    cuTry(cudaMalloc((void **) &d_exp, 2 * natoms * sizeof(float)));

    // Initialization
    init_rf_pulse<<<gridsize, blocksize, 0, stream[1]>>>(d_w, d_params,
        natoms, nparams, 180.f, 0.f);
    cudaDeviceSynchronize();

    // Begin iteration
    for (int i = 0; i < nreps; i++)
    {
        // Apply RF Pulse
        fill_transition_matrix<<<gridsize, blocksize, 0,
            stream[1]>>>(d_T_m, d_mrf, nreps, i);
        cudaDeviceSynchronize();

        apply_rf_pulse(d_w, d_T_m, natoms);

        // decay at TR
        compute_exp<<<gridsize, blocksize>>>(d_exp, d_params, d_mrf,
            nreps, natoms, nparams, i, 4);
        cudaDeviceSynchronize();
        decay_signal<<<gridsize, blocksize>>>(d_w, d_exp, natoms);
        cudaDeviceSynchronize();

        // shift phase at TR/2
        fill_offresoance_operator<<<gridsize, blocksize, 0,
            stream[1]>>>(d_rbeta, d_mrf, d_params, natoms, nreps, nparams, i);
        cudaDeviceSynchronize();
        shift_phase<<<gridsize, blocksize, 0, stream[1]>>>(d_w,
            d_rbeta, natoms);
        cudaDeviceSynchronize();

        // save atoms
        save_atom<<<gridsize, blocksize, 0, stream[1]>>>(d_atoms, d_w,
            nreps, natoms, i);
        cudaDeviceSynchronize();

        // shift phase at TR
        shift_phase<<<gridsize, blocksize, 0, stream[1]>>>(d_w,
            d_rbeta, natoms);
        cudaDeviceSynchronize();
    }

    cuTry(cudaFree(d_T_m));
    cuTry(cudaFree(d_rbeta));
    cuTry(cudaFree(d_w));
    cuTry(cudaFree(d_exp));
}

void
MRF_dict(float2 * d_atoms, float *d_params, float *h_mrf,
    size_t nreps, size_t NREPS, size_t NATOMS, size_t natoms,
    int nparams, int nstates, const char *echo_type)
{
    // Copy data from host to device
    float *d_mrf;
    cuTry(cudaMalloc((void **) &d_mrf, nreps * 4 * sizeof(float)));

    for (int i = 0; i < 4; i++)
        cuTry(cudaMemcpyAsync(d_mrf + i * nreps, h_mrf + i * NREPS,
                nreps * sizeof(float), cudaMemcpyHostToDevice, stream[0]));

    // Begin contruct dictionary
    printf("Constructing dictionary...\n");

    if (strcmp(echo_type, "epg_ssfp") == 0)
    {
        float TI = 40.f;
        float nominal_flip = 60.f;
        float TR_base = 16.f;
        float TE_base = 3.5;

        float *d_FA, *d_phi, *d_TR, *d_TE;

        d_FA = d_mrf;
        d_phi = d_mrf + nreps;
        d_TR = d_mrf + 2 * nreps;
        d_TE = d_mrf + 3 * nreps;

        arrayscale<<<gridsize, blocksize, 0, stream[0]>>>(d_FA, nreps,
            nominal_flip * M_PI / 180.f);
        arrayscale<<<gridsize, blocksize, 0, stream[0]>>>(d_phi, nreps,
            M_PI / 180.f);
        arrayadd<<<gridsize, blocksize, 0, stream[0]>>>(d_TR, nreps,
            TR_base);
        arrayadd<<<gridsize, blocksize, 0, stream[0]>>>(d_TE, nreps,
            TE_base);

        epg_ssfp(d_atoms, d_params, d_mrf, TI, nreps, NATOMS, natoms,
            nparams, nstates);
        cudaDeviceSynchronize();

        // normalize atoms to unit norm
        normalize<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms,
            nreps, NATOMS);
    }
    else if (strcmp(echo_type, "roa_ssfp") == 0)
    {
        float nominal_flip = 60.f;
        float TR_base = 16.f;
        float TE_base = 3.5;

        float *d_FA, *d_phi, *d_TR, *d_TE;

        d_FA = d_mrf;
        d_phi = d_mrf + nreps;
        d_TR = d_mrf + 2 * nreps;
        d_TE = d_mrf + 3 * nreps;

        arrayscale<<<gridsize, blocksize, 0, stream[0]>>>(d_FA, nreps,
            nominal_flip * M_PI / 180.f);
        arrayscale<<<gridsize, blocksize, 0, stream[0]>>>(d_phi, nreps,
            M_PI / 180.f);
        arrayadd<<<gridsize, blocksize, 0, stream[0]>>>(d_TR, nreps,
            TR_base);
        arrayadd<<<gridsize, blocksize, 0, stream[0]>>>(d_TE, nreps,
            TE_base);

        roa_ssfp(d_atoms, d_mrf, d_params, nreps, natoms, nparams);
        cudaDeviceSynchronize();

        // normalize atoms to unit norm
        normalize<<<gridsize, blocksize, 0, stream[0]>>>(d_atoms,
            nreps, natoms);
    }
    else if (strcmp(echo_type, "epg_sr") == 0)
        epg_sr(d_atoms, d_params, d_mrf, nreps, natoms, nparams, nstates);
    else if (strcmp(echo_type, "epg_se") == 0)
        epg_se(d_atoms, d_params, d_mrf, nreps, natoms, nparams, nstates);
    else if ((strcmp(echo_type, "epg_tse1") == 0)
        || (strcmp(echo_type, "epg_tse2") == 0))
        epg_tse(d_atoms, d_params, d_mrf, nreps, natoms, nparams, nstates);
    else if (strcmp(echo_type, "epg_fisp") == 0)
        epg_fisp(d_atoms, d_params, d_mrf, nreps, natoms, nparams,
            nstates);
    else
        err(EX_USAGE, "Please put in correct echo types.\n");

    // Free memory
    cuTry(cudaFree(d_mrf));
}

//// Macthing
__global__ void
generate_maps(float *d_maps, float2 * d_mat, float *d_params,
    size_t natoms, size_t nvoxels, int nparams, int nmaps)
{
    size_t matchid;
    float ipmax, idabs;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nvoxels;
        id += blockDim.x * gridDim.x)
    {
        matchid = 0;
        ipmax = 0.f;
        idabs = 0.f;

        for (int i = 0; i < natoms; i++)
        {
            idabs = abs(d_mat[id * natoms + i]);

            if (idabs > ipmax)
            {
                ipmax = idabs;
                matchid = i;
            }
        }

        d_maps[id * nmaps] = d_params[matchid * nparams];
        d_maps[id * nmaps + 1] = d_params[matchid * nparams + 1];
        d_maps[id * nmaps + 2] = d_params[matchid * nparams + 2];
        d_maps[id * nmaps + 3] = d_params[matchid * nparams + 3];
        d_maps[id * nmaps + 4] = ipmax;
    }
}

__global__ void
merge_maps(float *d_maps, float *d_MAPS, float *d_b1map,
    size_t NATOMS, size_t natoms, size_t nvoxels, int nmaps)
{
    size_t matchid;
    float b1_gt, b1, b1_diff, b1_abs;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nvoxels;
        id += blockDim.x * gridDim.x)
    {
        matchid = 0;
        b1_diff = 0.f;
        b1_abs = 10.f;
        b1_gt = d_b1map[id];

        for (int i = 0; i < NATOMS / natoms; i++)
        {
            b1 = d_MAPS[id * nmaps + i * nmaps * nvoxels + 3];
            b1_diff = abs(b1_gt - b1);

            if (b1_diff < b1_abs)
            {
                b1_abs = b1_diff;
                matchid = i;
            }
        }

        d_maps[id * nmaps] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels];
        d_maps[id * nmaps + 1] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 1];
        d_maps[id * nmaps + 2] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 2];
        d_maps[id * nmaps + 3] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 3];
        d_maps[id * nmaps + 4] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 4];
    }
}

__global__ void
merge_maps(float *d_maps, float *d_MAPS,
    size_t NATOMS, size_t natoms, size_t nvoxels, int nmaps)
{
    size_t matchid;
    float pd, pdmax;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < nvoxels;
        id += blockDim.x * gridDim.x)
    {
        matchid = 0;
        pdmax = 0.f;

        for (int i = 0; i < NATOMS / natoms; i++)
        {
            pd = d_MAPS[id * nmaps + i * nmaps * nvoxels + 4];

            if (pd > pdmax)
            {
                pdmax = pd;
                matchid = i;
            }
        }

        d_maps[id * nmaps] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels];
        d_maps[id * nmaps + 1] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 1];
        d_maps[id * nmaps + 2] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 2];
        d_maps[id * nmaps + 3] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 3];
        d_maps[id * nmaps + 4] = d_MAPS[id * nmaps + matchid * nmaps * nvoxels + 4];
    }
}

void
compute_nsplits(size_t * nsplits, size_t * nminivoxels,
    size_t nvoxels, size_t NATOMS, int type)
{
    // Compute available GPU memory
    size_t free_mem, total;
    cudaMemGetInfo(&free_mem, &total);
    printf("Available GPU memory = %lu MB\n", free_mem / 1024 / 1024);

    size_t max_float2 = free_mem / sizeof(float2) / 1.01;

    if (type == 1)
    {
        // Split data into uneven parts to avoid out-of-memory
        *nminivoxels = min(max_float2 / NATOMS, nvoxels);
        *nsplits = nvoxels / (*nminivoxels);
    }
    else if (type == 0)
    {
        //Split data into even parts to avoid out-of-memory
        if (NATOMS * nvoxels > max_float2)
        {
            for (int i = 2; i < nvoxels / 2 + 1; i++)
            {
                if (nvoxels % i == 0)
                {
                    *nsplits = i;
                    *nminivoxels = nvoxels / (*nsplits);

                    if (NATOMS * (*nminivoxels) < max_float2)
                        break;
                }
            }
        }
        else
        {
            *nminivoxels = nvoxels;
            *nsplits = 1;
        }
    }
    else
    {
        printf("Please put in 0 or 1\n");
    }
}

void
MRF_minimatch(float *d_maps, float2 * d_img, float2 * d_atoms,
    float2 * d_mat, float *d_params, size_t nreps, size_t natoms,
    size_t nminivoxels, int nparams, int nmaps)
{
    // Matching data using build-in cublas function
    // Matching nminivoxels voxels at a time

    float2 alpha = make_float2(1.f);
    float2 beta = make_float2(0.f);

    // Create handle
    stat = cublasCreate(&handle);

    cublasCgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, natoms, nminivoxels,
        nreps, &alpha, d_atoms, nreps, d_img, nreps, &beta, d_mat, natoms);

    if (stat != CUBLAS_STATUS_SUCCESS)
        err(EX_SOFTWARE, "CUBLAS Cgemm failed:%d\n", stat);

    // Select the maximum absolute value for each column
    generate_maps<<<gridsize, blocksize, 0, stream[1]>>>(d_maps, d_mat,
        d_params, natoms, nminivoxels, nparams, nmaps);
    cudaDeviceSynchronize();

    // Destory handle
    cublasDestroy(handle);
}

void
MRF_match(float *d_maps, float2 *h_img, float2 *d_atoms, float *d_params,
    size_t NREPS, size_t nreps, size_t NATOMS, size_t natoms, size_t nvoxels,
    int nparams, int nmaps, const char *in_b1map)
{
    int l_b1 = NATOMS / natoms;
    float *d_MAPS;
    cuTry(cudaMalloc((void **) &d_MAPS,
              nmaps * nvoxels * l_b1 * sizeof(float)));

    // Compute nsplits and nminivoxels
    size_t nsplits, nminivoxels;
    int type = 0;
    compute_nsplits(&nsplits, &nminivoxels, nvoxels, nreps + natoms, type);
    printf("nsplits = %lu, nminivoxels = %lu\n", nsplits, nminivoxels);

    // Malloc device arrays
    float2 *d_img;
    cuTry(cudaMalloc((void **) &d_img,
            nreps * nminivoxels * sizeof(float2)));

    float2 *d_mat;
    cuTry(cudaMalloc((void **) &d_mat,
            natoms * nminivoxels * sizeof(float2)));

    printf("Matching...\n");

    for (int k = 0; k < l_b1; k++)
    {
        for (int i = 0; i < nsplits; i++)
        {
            for (int j = 0; j < nminivoxels; j++)
            {
                cuTry(cudaMemcpyAsync(d_img + j * nreps,
                        h_img + i * NREPS * nminivoxels + j * NREPS,
                        nreps * sizeof(float2), cudaMemcpyHostToDevice,
                        stream[1]));
            }

            MRF_minimatch(d_MAPS + k * nmaps * nvoxels + i * nmaps * nminivoxels,
                d_img, d_atoms + k * nreps * natoms, d_mat,
                d_params + k * nparams * natoms,
                nreps, natoms, nminivoxels, nparams, nmaps);
        }

        if (type == 1)
        {
            for (int i = 0; i < nvoxels - nsplits * nminivoxels; i++)
            {
                cuTry(cudaMemcpyAsync(d_img + i * nreps,
                        h_img + nsplits * NREPS * nminivoxels + i * NREPS,
                        nreps * sizeof(float2), cudaMemcpyHostToDevice,
                        stream[1]));
            }

            MRF_minimatch(d_MAPS + k * nmaps * nvoxels + nsplits * nmaps * nminivoxels,
                d_img, d_atoms + k * nreps * natoms, d_mat,
                d_params + k * nparams * natoms,
                nreps, natoms, (nvoxels - nsplits * nminivoxels), nparams, nmaps);
        }
      }

      // Merge submaps into one map
      float *d_b1map;
      if (in_b1map != NULL)
      {
          printf("Input B1 map: %s\n", in_b1map);
          ra_t ra_b1map;
          printf("Reading %s\n", in_b1map);
          ra_read(&ra_b1map, in_b1map);

          float *h_b1map = (float *) ra_b1map.data;
          cuTry(cudaMalloc((void **) &d_b1map, nvoxels * sizeof(float)));
          cuTry(cudaMemcpyAsync(d_b1map, h_b1map, nvoxels * sizeof(float),
                  cudaMemcpyHostToDevice, stream[1]));

          merge_maps<<<gridsize, blocksize, 0, stream[0]>>>(d_maps, d_MAPS, d_b1map,
              NATOMS, natoms, nvoxels, nmaps);
          cudaDeviceSynchronize();

          ra_free(&ra_b1map);
          cuTry(cudaFree(d_b1map));
        }
        else
            merge_maps<<<gridsize, blocksize, 0, stream[0]>>>(d_maps, d_MAPS,
                NATOMS, natoms, nvoxels, nmaps);
            cudaDeviceSynchronize();

        cuTry(cudaFree(d_img));
        cuTry(cudaFree(d_mat));
        cuTry(cudaFree(d_MAPS));
}


//// Unittest
void
generate_echo(float *h_echo, size_t nreps, IP ip)
{
    for (int i = 0; i < nreps; i++)
    {
        h_echo[0 * nreps + i] = ip.alpha;
        h_echo[1 * nreps + i] = ip.phi;
        h_echo[2 * nreps + i] = ip.TR;
        h_echo[3 * nreps + i] = ip.TE;
    }
}

void
compare(int *pass, int *fail, float2 * d_atoms, size_t nreps,
    size_t NATOMS, const char *echo_type)
{
    ra_t ra_atoms_gt;
    float2 *h_atoms, *h_atoms_gt;

    char filepath_gt[1024];
    snprintf(filepath_gt, 1024, "../data/atoms_%s.ra", echo_type);

    ra_read(&ra_atoms_gt, filepath_gt);
    h_atoms_gt = (float2 *) ra_atoms_gt.data;

    h_atoms = (float2 *) safe_malloc(nreps * NATOMS * sizeof(float2));
    cuTry(cudaMemcpyAsync(h_atoms, d_atoms,
            nreps * NATOMS * sizeof(float2), cudaMemcpyDeviceToHost));

    float diff = 0.f;
    int stride = 0;
    for (int i = 0; i < NATOMS; i++)
    {
        printf("The %d th atom\n", i + 1);

        stride = i * nreps;
        for (int j = 0; j < nreps; j++)
        {
            printf("atoms_%s[%d] = %f + %fim, ", echo_type, j,
                h_atoms[j + stride].x, h_atoms[j + stride].y);
            printf("atoms_gt[%d] = %f + %fim\n", j,
                h_atoms_gt[j + stride].x, h_atoms_gt[j + stride].y);
            diff += abs(h_atoms[j + stride] - h_atoms_gt[j + stride]);
        }
    }

    if (diff < 1E-6)
    {
        printf("%s unit test passed!\n", echo_type);
        printf("\n");
        (*pass)++;
    }
    else
    {
        printf("%s unit test failed!\n", echo_type);
        printf("\n");
        (*fail)++;
    }

    ra_free(&ra_atoms_gt);
    safe_free(h_atoms);
}

void
unittest_dictionary(int *pass, int *fail, float *h_echo,
    size_t nreps, size_t NREPS, int nparams, int nstates,
    const char *T1, const char *T2, const char *B0,
    const char *B1, const char *echo_type)
{
    // Parse parameters
    size_t l_t1, l_t2, l_b0, l_b1, NATOMS, natoms;

    l_t1 = parse_length(T1);
    float *h_t1 = (float *) safe_malloc(l_t1 * sizeof(float));
    parse_params(h_t1, T1, l_t1);

    l_t2 = parse_length(T2);
    float *h_t2 = (float *) safe_malloc(l_t2 * sizeof(float));
    parse_params(h_t2, T2, l_t2);

    l_b0 = parse_length(B0);
    float *h_b0 = (float *) safe_malloc(l_b0 * sizeof(float));
    parse_params(h_b0, B0, l_b0);

    l_b1 = parse_length(B1);
    float *h_b1 = (float *) safe_malloc(l_b1 * sizeof(float));
    parse_params(h_b1, B1, l_b1);

    // Compute the number of atoms
    NATOMS = compute_natoms(h_t1, h_t2, l_t1, l_t2, l_b0, l_b1);
    natoms = NATOMS/l_b1;

    // Transfer d_t1, d_t2 and d_b0 into d_params;
    float *h_params =
        (float *) safe_malloc(NATOMS * nparams * sizeof(float));
    trans_params(h_params, h_t1, h_t2, h_b0, h_b1,
        l_t1, l_t2, l_b0, l_b1, NATOMS, nparams);

    float *d_params;
    cuTry(cudaMalloc((void **) &d_params,
            nparams * NATOMS * sizeof(float)));
    cuTry(cudaMemcpyAsync(d_params, h_params,
            nparams * NATOMS * sizeof(float), cudaMemcpyHostToDevice,
            stream[1]));

    safe_free(h_t1);
    safe_free(h_t2);
    safe_free(h_b0);
    safe_free(h_b1);
    safe_free(h_params);

    float2 *d_atoms;
    cuTry(cudaMalloc((void **) &d_atoms, nreps * NATOMS * sizeof(float2)));

    printf("Begin unit test for %s\n", echo_type);
    printf("T1 = %s\n", T1);
    printf("T2 = %s\n", T2);
    printf("B0 = %s\n", B0);
    printf("B1 = %s\n", B1);
    printf("l_t1: %lu, l_t2: %lu, l_b0: %lu, l_b1: %lu\n",
        l_t1, l_t2, l_b0, l_b1);
    printf("nreps: %lu, NATOMS: %lu, natoms: %lu\n", nreps, NATOMS, natoms);

    MRF_dict(d_atoms, d_params, h_echo, nreps, NREPS, NATOMS, natoms,
        nparams, nstates, echo_type);

    compare(pass, fail, d_atoms, nreps, NATOMS, echo_type);

    cuTry(cudaFree(d_atoms));
    cuTry(cudaFree(d_params));
}

void
unittest_matching(int *pass, int *fail,
    const char *atoms_path, const char *params_path,
    int nparams, int nmaps, int ntests)
{
    // Read atoms and corresponding paramseters and copy them to the device
    ra_t ra_atoms, ra_params;
    float2 *h_atoms, *d_atoms;
    float *h_params, *h_maps, *d_params, *d_maps;
    size_t nreps, NREPS, natoms, NATOMS, nvoxels;
    const char *in_b1map = NULL;
    int index[1024];

    printf("Reading %s\n", atoms_path);
    ra_read(&ra_atoms, atoms_path);
    h_atoms = (float2 *) ra_atoms.data;
    NREPS = ra_atoms.dims[0];
    NATOMS = ra_atoms.dims[1];
    natoms = NATOMS;
    nreps = NREPS;
    nvoxels = ntests;
    printf("NREPS: %lu, NATOMS: %lu\n", NREPS, NATOMS);
    cuTry(cudaMalloc((void **) &d_atoms, NREPS * NATOMS * sizeof(float2)));
    cuTry(cudaMemcpyAsync(d_atoms, h_atoms,
            NREPS * NATOMS * sizeof(float2), cudaMemcpyHostToDevice));

    printf("Reading %s\n", params_path);
    ra_read(&ra_params, params_path);
    h_params = (float *) ra_params.data;

    cuTry(cudaMalloc((void **) &d_params,
            nparams * NATOMS * sizeof(float)));
    cuTry(cudaMemcpyAsync(d_params, h_params,
            nparams * NATOMS * sizeof(float), cudaMemcpyHostToDevice));

    float2 *h_img = (float2 *) safe_malloc(NREPS * nvoxels * sizeof(float2));

    for (int i = 0; i < ntests; i++)
    {
        index[i] = rand() % NATOMS;
        for (int j = 0; j < NREPS; j++)
        {
            h_img[i*NREPS+j] = h_atoms[index[i]*NREPS+j];
        }
    }

    // Run matching
    cuTry(cudaMalloc((void **) &d_maps, nmaps * ntests * sizeof(float)));

    MRF_match(d_maps, h_img, d_atoms, d_params, NREPS,
        nreps, NATOMS, natoms, nvoxels, nparams, nmaps, in_b1map);

    h_maps = (float *) safe_malloc(nmaps * ntests * sizeof(float));
    cuTry(cudaMemcpyAsync(h_maps, d_maps, nmaps * ntests * sizeof(float),
            cudaMemcpyDeviceToHost));

    // Compare results
    float diff = 0.f;
    for (int i = 0; i < ntests; i++)
    {
        printf("The %dth voxel\n", i + 1);

        printf("T1_test[%d] = %.2f, ", i, h_maps[i * nmaps]);
        printf("T1_gt[%d] = %.2f\n", i, h_params[index[i] * nparams]);
        diff += abs(h_maps[i * nmaps] - h_params[index[i] * nparams]);

        printf("T2_test[%d] = %.2f, ", i, h_maps[i * nmaps + 1]);
        printf("T2_gt[%d] = %.2f\n", i, h_params[index[i] * nparams + 1]);
        diff +=
            abs(h_maps[i * nmaps + 1] - h_params[index[i] * nparams + 1]);

        printf("B0_test[%d] = %.2f, ", i, h_maps[i * nmaps + 2]);
        printf("B0_gt[%d] = %.2f\n", i, h_params[index[i] * nparams + 2]);
        diff +=
            abs(h_maps[i * nmaps + 2] - h_params[index[i] * nparams + 2]);

        printf("B1_test[%d] = %.2f, ", i, h_maps[i * nmaps + 3]);
        printf("B1_gt[%d] = %.2f\n", i, h_params[index[i] * nparams + 3]);
        diff +=
            abs(h_maps[i * nmaps + 3] - h_params[index[i] * nparams + 3]);
    }

    if (diff < 1E-6)
    {
        printf("Matching unit test passed!\n");
        printf("\n");
        (*pass)++;
    }
    else
    {
        printf("Matching unit test failed!\n");
        printf("\n");
        (*fail)++;
    }

    // Free memory
    ra_free(&ra_atoms);
    ra_free(&ra_params);

    safe_free(h_maps);

    cuTry(cudaFree(d_atoms));
    cuTry(cudaFree(d_params));
    cuTry(cudaFree(d_maps));
}
