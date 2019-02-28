/*
  This file is part of the MRF_CUDA package (https://github.com/chixindebaoyu/MRF_CUDA).

  The MIT License (MIT)

  Copyright (c) Dong Wang

  Permission is hereby granted, free of charge, to any person obtaining a # copy
  of this software and associated documentation files (the "Software"), to # deal
  in the Software without restriction, including without limitation the # rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or # sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included # in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS # OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING # FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS # IN THE
  SOFTWARE.
*/

#include "functions.h"

extern int blocksize;
extern int gridsize;

extern cudaStream_t stream[2];

int
main(int argc, char *argv[])
{
    // Setup CUDA
    cudaSetDevice(0);
    cuTry(cudaStreamCreate(&stream[0]));
    cuTry(cudaStreamCreate(&stream[1]));

    // Setup default values
    const char *in_mrf, *in_img, *in_b1map, *out_atoms, *out_maps, *out_params;
    in_mrf = "../data/MRF_100.csv";
    in_img = NULL;
    in_b1map = NULL;
    out_maps = NULL;
    out_atoms = NULL;
    out_params = NULL;

    const char *T1, *T2, *B0, *B1, *echo_type;
    T1 = "20:20:3000";
    T2 = "20:10:1000";
    B0 = "0";
    B1 = "1";
    echo_type = "epg_ssfp";

    int nparams = 4;
    int nmaps = nparams + 1;
    int nstates = 101;
    size_t nreps = 1500;
    size_t NATOMS = 0;
    size_t natoms = 0;

    // Read command line
    struct option long_options[] =
    {
        {"b1map", 1, NULL, 'M'},
        {"atoms", 1, NULL, 'a'},
        {"map", 1, NULL, 'm'},
        {"params", 1, NULL, 'p'},
        {"T1", 1, NULL, 't'},
        {"T2", 1, NULL, 's'},
        {"B0", 1, NULL, 'b'},
        {"B1", 1, NULL, 'r'},
        {"nstates", 1, NULL, 'w'},
        {"echo_type", 1, NULL, 'e'},
        {"nreps", 1, NULL, 'T'},
        {"gridsize", 1, NULL, 'G'},
        {"blocksize", 1, NULL, 'B'},
        {"help", 0, 0, 'h'}
    };

    extern int optind;
    opterr = 0;
    int option_index = 0;
    int c;
    while ((c =
            getopt_long(argc, argv, "M:N:a:m:p:t:s:b:r:w:e:T:G:B:h",
                long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'M':
                in_b1map = optarg;
                break;
            case 'N':
                NATOMS = atoi(optarg);
                break;
            case 'a':
                out_atoms = optarg;
                break;
            case 'm':
                out_maps = optarg;
                break;
            case 'p':
                out_params = optarg;
                break;
            case 't':
                T1 = optarg;
                break;
            case 's':
                T2 = optarg;
                break;
            case 'b':
                B0 = optarg;
                break;
            case 'r':
                B1 = optarg;
                break;
            case 'w':
                nstates = atoi(optarg);
                break;
            case 'e':
                echo_type = optarg;
                break;
            case 'T':
                nreps = atoi(optarg);
                break;
            case 'G':
                gridsize = atoi(optarg);
                break;
            case 'B':
                blocksize = atoi(optarg);
                break;
            case 'h':
            default:
                print_usage();
                return 1;
        }
    }

    int csv_count = 0;
    int ra_count = 0;
    const char *tmp;
    while (optind <= argc-1)
    {
        tmp = argv[optind];

        if ((strstr(tmp,".ra") != NULL) && (ra_count == 0))
        {
            in_img = tmp;
            ra_count ++;
        }

        if ((strstr(tmp,".csv") != NULL) && (csv_count == 0))
        {
            in_mrf = tmp;
            csv_count ++;
        }

        if (csv_count+ra_count == 2)
            break;

        optind ++;
    }

    printf("\n");
    printf("nstates : %d\n", nstates);
    printf("Blocksize : %d\n", blocksize);
    printf("Gridsize : %d\n", gridsize);
    printf("Echo : %s\n", echo_type);
    printf("\n");

    // Parse parameters
    size_t l_t1, l_t2, l_b0, l_b1;
    float *h_t1, *h_t2, *h_b0, *h_b1;

    if (NATOMS != 0)
    {
        printf("Hard coding for parameters...\n");

        l_b1 = 11;
        l_b0 = 21;
        l_t1 = roundf(sqrtf(NATOMS / l_b0 / l_b1));

        if (l_t1 == 0)
            l_t1 = 1;

        l_t2 = l_t1;

        h_t1 = (float *) safe_malloc(l_t1 * sizeof(float));
        logspace(h_t1, 100.f, 3000.f, l_t1);

        h_t2 = (float *) safe_malloc(l_t2 * sizeof(float));
        logspace(h_t2, 20.f, 1000.f, l_t2);

        h_b0 = (float *) safe_malloc(l_b0 * sizeof(float));
        logspace(h_b0, -150.f, 150.f, l_b0);

        h_b1 = (float *) safe_malloc(l_b1 * sizeof(float));
        logspace(h_b1, 0.5, 1.5, l_b1);
    }
    else
    {
        printf("Parsing parameters...\n");
        printf("T1 : %s\n", T1);
        printf("T2 : %s\n", T2);
        printf("B0 : %s\n", B0);
        printf("B1 : %s\n", B1);

        l_t1 = parse_length(T1);
        h_t1 = (float *) safe_malloc(l_t1 * sizeof(float));
        parse_params(h_t1, T1, l_t1);

        l_t2 = parse_length(T2);
        h_t2 = (float *) safe_malloc(l_t2 * sizeof(float));
        parse_params(h_t2, T2, l_t2);

        l_b0 = parse_length(B0);
        h_b0 = (float *) safe_malloc(l_b0 * sizeof(float));
        parse_params(h_b0, B0, l_b0);

        l_b1 = parse_length(B1);
        h_b1 = (float *) safe_malloc(l_b1 * sizeof(float));
        parse_params(h_b1, B1, l_b1);
      }

    // The number of atoms of the whole dictionary
    NATOMS = compute_natoms(h_t1, h_t2, l_t1, l_t2, l_b0, l_b1);
    // The number of atoms per b1
    natoms = NATOMS/l_b1;

    printf("l_t1: %lu, l_t2: %lu, l_b0: %lu, l_b1: %lu\n",
        l_t1, l_t2, l_b0, l_b1);
    printf("NATOMS: %lu, natoms: %lu\n", NATOMS, natoms);
    printf("Removing situations when T1 <= T2\n");
    printf("NATOMS might be smaller than input\n");

    if (NATOMS == 0)
        err(EX_SOFTWARE, "Number of atoms is 0!\n");
    printf("\n");

    // Transfer h_t1, h_t2, h_b0 and h_b1 into h_params;
    float *h_params =
        (float *) safe_malloc(NATOMS * nparams * sizeof(float));
    trans_params(h_params, h_t1, h_t2, h_b0, h_b1, l_t1, l_t2, l_b0, l_b1,
      NATOMS, nparams);

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

    // Read input mrf
    printf("Input mrf: %s\n", in_mrf);
    printf("Reading %s\n", in_mrf);
    size_t NREPS = csv_dim(in_mrf) - 1;

    if (nreps > NREPS)
        nreps = NREPS;

    printf("NREPS: %lu, nreps: %lu\n", NREPS, nreps);

    float *h_mrf = (float *) safe_malloc(NREPS * 4 * sizeof(float));
    csv_read(h_mrf, NREPS, in_mrf);

    printf("\n");

    // Run MRF Dictionary
    float2 *d_atoms;
    cuTry(cudaMalloc((void **) &d_atoms,
            nreps * NATOMS * sizeof(float2)));

    clock_t start, stop;

    start = clock();
    MRF_dict(d_atoms, d_params, h_mrf, nreps, NREPS, NATOMS, natoms,
        nparams, nstates, echo_type);
    cudaDeviceSynchronize();
    stop = clock();

    printf("Elapsed dictionary time: %.2f s\n",
        ((float) (stop - start)) / CLOCKS_PER_SEC);
    printf("\n");

    safe_free(h_mrf);

    // Run matching if in_img is specified
    if ((in_img == NULL) && (out_maps != NULL))
        printf("NO data specified! Abort matching!\n");

    if (in_img != NULL)
    {
        printf("Input img: %s\n", in_img);
        ra_t ra_img;
        printf("Reading %s\n", in_img);
        ra_read(&ra_img, in_img);

        float2 *h_img = (float2 *) ra_img.data;
        size_t nt = ra_img.dims[0];
        size_t nvoxels = ra_img.dims[1] * ra_img.dims[2];
        printf("nt: %lu, nx: %llu, ny: %llu\n", nt, ra_img.dims[1],
            ra_img.dims[2]);
        printf("\n");

        // Run MRF Macthing
        float *d_maps;
        cuTry(cudaMalloc((void **) &d_maps,
                nmaps * nvoxels * sizeof(float)));

        start = clock();
        MRF_match(d_maps, h_img, d_atoms, d_params,
            nreps, nt, NATOMS, natoms, nvoxels, nparams, nmaps, in_b1map);
        cudaDeviceSynchronize();
        stop = clock();

        printf("Elapsed matching time: %.2f s\n",
            ((float) (stop - start)) / CLOCKS_PER_SEC);
        printf("\n");

        // Save maps
        if (out_maps == NULL)
            out_maps = "../result/maps.ra";

        printf("Output maps: %s\n", out_maps);
        float *h_map;
        h_map = (float *) safe_malloc(nmaps * nvoxels * sizeof(float));
        cuTry(cudaMemcpyAsync(h_map, d_maps,
                nmaps * nvoxels * sizeof(float),
                cudaMemcpyDeviceToHost, stream[1]));
        save_rafile(h_map, out_maps, nmaps, nvoxels);

        safe_free(h_map);
        ra_free(&ra_img);
        cuTry(cudaFree(d_maps));
    }

    // Save atoms
    if (out_atoms != NULL)
    {
        printf("Output atoms: %s\n", out_atoms);
        float2 *h_atoms;
        h_atoms = (float2 *) safe_malloc(nreps * NATOMS * sizeof(float2));
        cuTry(cudaMemcpyAsync(h_atoms, d_atoms,
                nreps * NATOMS * sizeof(float2), cudaMemcpyDeviceToHost,
                stream[0]));
        save_rafile(h_atoms, out_atoms, nreps, NATOMS);

        safe_free(h_atoms);
        cuTry(cudaFree(d_atoms));
    }
    else
        cuTry(cudaFree(d_atoms));

    // Save parameters
    if (out_params != NULL)
    {
        printf("Output params: %s\n", out_params);
        float *h_params;
        h_params = (float *) safe_malloc(nparams * NATOMS * sizeof(float));
        cuTry(cudaMemcpyAsync(h_params, d_params,
                nparams * NATOMS * sizeof(float), cudaMemcpyDeviceToHost,
                stream[0]));
        save_rafile(h_params, out_params, nparams, NATOMS);

        safe_free(h_params);
        cuTry(cudaFree(d_params));
    }
    else
        cuTry(cudaFree(d_params));

    // Destory cuda stream
    cuTry(cudaStreamDestroy(stream[0]));
    cuTry(cudaStreamDestroy(stream[1]));

    return 0;
}
