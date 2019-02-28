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

int
main(int argc, char *argv[])
{
  int pass = 0, fail = 0;
  size_t nreps, NREPS;
  int nparams = 4;
  int nmaps = nparams+1;
  int nstates = 101;
  IP ip;
  const char *T1, *T2, *B0, *B1;

  // Unit test for spin echo
  // 90-180
  nreps = 1;
  NREPS = 1;
  ip.alpha = M_PI;
  ip.phi = 0.f;
  ip.TR = 1000.f;
  ip.TE = 25.f;

  float *h_epg_se = (float*)safe_malloc(nreps*4*sizeof(float));

  generate_echo(h_epg_se, nreps, ip);

  T1 = "600:600:600";
  T2 = "100:100:100";
  B0 = "0:1:0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_epg_se, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "epg_se");

  safe_free(h_epg_se);

  // Unit test for saturation recovery
  // 60-60-60-...
  nreps = 10;
  NREPS = 10;
  ip.alpha = M_PI/3;
  ip.phi = M_PI/2;
  ip.TR = 500.f;
  ip.TE = 1.f;

  float *h_epg_sr = (float*)safe_malloc(nreps*4*sizeof(float));

  generate_echo(h_epg_sr, nreps, ip);

  T1 = "600";
  T2 = "100";
  B0 = "0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_epg_sr, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "epg_sr");

  safe_free(h_epg_sr);

  // Unit test for turbo spin echo
  // 90-120-120-... with no relaxation
  nreps = 3;
  NREPS = 3;
  ip.alpha = 2*M_PI/3;
  ip.phi = 0.f;
  ip.TR = 0.f; // correpond to T1 and T2 are infinity
  ip.TE = 0.f; // correpond to T1 and T2 are infinity

  float *h_epg_tse1 = (float*)safe_malloc(nreps*4*sizeof(float));

  generate_echo(h_epg_tse1, nreps, ip);

  T1 = "1000:1000:1000";
  T2 = "100:100:100";
  B0 = "0:1:0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_epg_tse1, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "epg_tse1");

  safe_free(h_epg_tse1);

  // Unit test for turbo spin echo
  // 90-180-180-... with relaxation
  nreps = 10;
  NREPS = 10;
  ip.alpha = M_PI;
  ip.phi = 0.f;
  ip.TR = 50.f;
  ip.TE = 25.f;

  float *h_epg_tse2 = (float*)safe_malloc(nreps*4*sizeof(float));

  generate_echo(h_epg_tse2, nreps, ip);

  T1 = "600:600:600";
  T2 = "100:100:100";
  B0 = "0:1:0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_epg_tse2, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "epg_tse2");

  safe_free(h_epg_tse2);

  // Unit test for F0 type SSFP (gradient spoiling but no RF spoiling)
  // 30-30-30-....
  nreps = 3;
  NREPS = 3;
  ip.alpha = M_PI/6;
  ip.phi = 0.f;
  ip.TR = 10.f;
  ip.TE = 5.f;

  float *h_epg_fisp = (float*)safe_malloc(nreps*4*sizeof(float));

  generate_echo(h_epg_fisp, nreps, ip);

  T1 = "1000:1000";
  T2 = "100:100";
  B0 = "0:0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_epg_fisp, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "epg_fisp");

  safe_free(h_epg_fisp);

  // Unit test for real MRF bSSFP sequence using EPG
  // Flip angle and phase are contained in the csv file
  const char *in_epg_ssfp = "../data/MRF_5.csv";
  printf("Reading %s\n", in_epg_ssfp);
  NREPS = csv_dim(in_epg_ssfp) - 1;
  nreps = NREPS;
  float *h_epg_ssfp = (float*)safe_malloc(nreps*4*sizeof(float));
  csv_read(h_epg_ssfp, nreps, in_epg_ssfp);

  T1 = "100:100:200";
  T2 = "20+20:20:40+1000";
  B0 = "0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_epg_ssfp, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "epg_ssfp");

  safe_free(h_epg_ssfp);

  // Unit test for Matching
  // Randomly select ntests atoms from a predifined dictionary as image voxels
  // Then match them to the dictionary
  const char *atoms_path = "../data/atoms_mrf.ra";
  const char *params_path = "../data/params_mrf.ra";

  int ntests = 10;
  unittest_matching(&pass, &fail, atoms_path, params_path, nparams, nmaps, ntests);

  // Unit test for real MRF bSSFP sequence using ROA
  // Flip angle and phase are contained in the csv file
  const char *in_roa_ssfp = "../data/MRF_5.csv";
  printf("Reading %s\n", in_roa_ssfp);
  NREPS = csv_dim(in_roa_ssfp) - 1;
  nreps = NREPS;
  float *h_roa_ssfp = (float*)safe_malloc(nreps*4*sizeof(float));
  csv_read(h_roa_ssfp, nreps, in_roa_ssfp);

  T1 = "100:100:200";
  T2 = "20+20:20:40+1000";
  B0 = "0";
  B1 = "1";

  unittest_dictionary(&pass, &fail, h_roa_ssfp, nreps, NREPS, nparams, nstates,
      T1, T2, B0, B1, "roa_ssfp");

  safe_free(h_roa_ssfp);

  // Print results
  printf("%d tests, %d passed and %d failed.\n", pass+fail, pass, fail);

  return 0;
}
