%% run example MRF EPG uSSFP generation
clear;

%% generate sequence params
csvdat = csvread('../data/MRF_100.csv',1,0);

nreps = size( csvdat, 1 );
nominal_flip = 60;
TR_base = 16;
TE_base = 3.5;
TI = 40;

FA_v = nominal_flip*csvdat(1:nreps,1);
phi_v = csvdat(1:nreps,2);
TR_v = TR_base + csvdat(1:nreps,3);
TE_v = TE_base + csvdat(1:nreps,4);

delk = 1;
szomega = 101;

T1 = unique([10:10:100,100:10:200]);
T2 = unique([4:2:20]);
B0 = unique([0]);

natoms = 0;

for i = 1:numel(T1)
  for j = 1:numel(T2)
    if T1(i) > T2(j)
      natoms = natoms + 1;
    end
  end
end

natoms = numel(B0)*natoms;
fprintf("natoms=%d\n",natoms);

atoms = zeros(nreps, natoms);
params = zeros(3, natoms);

tic;
index = 0;
for k = 1:numel(B0)
  for j = 1:numel(T2)
    for i = 1:numel(T1)
      if T1(i) > T2(j)
        index = index + 1;
        atoms(:,index) = EPG_MRF_SSFP(T1(i),T2(j),B0(k),TE_v,TR_v,FA_v,delk,nreps,szomega,phi_v,TI);
        atoms(:,index) = atoms(:,index)/norm(atoms(:,index));

        params(1,index) = T1(i);
        params(2,index) = T2(j);
        params(3,index) = B0(k);
      end
    end
  end
end

t = toc;
fprintf("dictionary time=%.2f\n",t);
% rawrite(atoms,"../result/atoms_matlab_epg.ra");
% rawrite(params,"../result/params_matlab_epg.ra");

%% Matching
img = raread('../data/phantom_100.ra');

[nreps,rows,cols] = size(img);
img = reshape(img,[nreps,rows*cols]);

%%
tic;
[proton, idx] = max(abs(atoms'*img));

maps = zeros(4,rows*cols);
for i = 1:rows*cols
    maps(1:3,i) = params(:,idx(i));
    maps(4,i) = proton(i);
end

t = toc;
fprintf("matching time=%.2f\n",t);

% rawrite(maps,"../result/maps_matlab_epg.ra");