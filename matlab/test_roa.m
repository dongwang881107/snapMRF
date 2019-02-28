%% run example MRF ROA uSSFP generation
clear;

%% read sequence params
csvdat = csvread('../data/MRF_100.csv',1,0);
img = raread('../data/phantom_100.ra');
B1_gt = raread('../data/B1_phantom_100.ra');

% csvdat = csvread('/data/MRF/MRF103.csv',1,0);
% img = raread('/data/MRF/phantom/image_varTE_103.ra');
% B1_gt = raread('/data/MRF/phantom/B1.ra');

% csvdat = csvread('/data/MRF/phantom/MRF001.csv',1,0);
% img = raread('/data/MRF/phantom/image_001.ra');
% B1_gt = raread('/data/MRF/phantom/B1.ra');

% csvdat = csvread('/data/MRF/brain/MRF001.csv',1,0);
% img = raread('../data/001_varTR.ra');
% B1_gt = raread('/data/MRF/brain/B1.ra');

nreps = size(csvdat,1);
nominal_flip = 60;
TR_base = 16;
TE_base = 3.5;

FA_v = nominal_flip*csvdat(1:nreps,1);
phi_v = csvdat(1:nreps,2);
TR_v = TR_base + csvdat(1:nreps,3);
TE_v = TE_base + csvdat(1:nreps,4);

%% generate test signal
% test
T1 = unique([10:10:100,100:10:200]);
T2 = unique([4:2:20]);
B0 = unique([0]);
B1 = unique([0.5:0.1:1.5]);

% phantom
% T1 = unique([50:5:200,200:25:500,500:100:3000]);
% T2 = unique([4:2:20,20:5:100,100:20:200,200:100:1000]);
% B0 = unique([-150:30:-90,-80:20:-40,-40:10:40,40:20:80,90:30:150]);
% B1 = unique([0.5:0.1:1.5]);
% path = '../result/maps_matlab_phantom_varTE.ra';
% path = '../result/maps_matlab_phantom_varTR.ra';

%brain
% T1 = unique([100:100:4000]);
% T2 = unique([20:20:2000]);
% B0 = unique([-400:50:400]);
% B1 = unique([0.5:0.1:1.5]);
% path = '../result/maps_matlab_brain_varTR.ra';

natoms = 0;

for i = 1:numel(T1)
  for j = 1:numel(T2)
    if T1(i) > T2(j)
      natoms = natoms + 1;
    end
  end
end

natoms_b1 = numel(B0)*natoms;
natoms = numel(B1)*natoms_b1;
fprintf('natoms=%d\n',natoms);

atoms = zeros(nreps, natoms);
params = zeros(4, natoms);

tic;
index = 0;
for r = 1:numel(B1)
  for k = 1:numel(B0)
    for j = 1:numel(T2)
      for i = 1:numel(T1)
        if T1(i) > T2(j)
          index = index + 1;
          FA_b1_v = FA_v*B1(r);
          atoms(:,index) = makeMRFdictionary(FA_v,phi_v,TR_v ,T1(i), T2(j), B0(k));
          atoms(:,index) = atoms(:,index)/norm(atoms(:,index));

          params(1,index) = T1(i);
          params(2,index) = T2(j);
          params(3,index) = B0(k);
          params(4,index) = B1(r);
        end
      end
    end
  end
end

t = toc;
fprintf('dictionary time=%.2f\n',t);
% rawrite(atoms,"../result/atoms_matlab_roa.ra");
% rawrite(params,"../result/params_matlab_roa.ra");

%% Matching
[nreps,rows,cols] = size(img);
img = reshape(img,[nreps,rows*cols]);

maps = zeros(5,rows*cols);
MAPS = zeros(5,rows*cols,numel(B1));

tic;
for k = 1:numel(B1)
  range = (k-1)*natoms_b1+1:k*natoms_b1;
  [proton,idx] = max(abs(atoms(:,range)'*img));
  for i = 1:rows*cols
    MAPS(1,i,k) = params(1,(k-1)*natoms_b1+idx(i));
    MAPS(2,i,k) = params(2,(k-1)*natoms_b1+idx(i));
    MAPS(3,i,k) = params(3,(k-1)*natoms_b1+idx(i));
    MAPS(4,i,k) = params(4,(k-1)*natoms_b1+idx(i));
    MAPS(5,i,k) = proton(i);
  end
end

B1_gt = B1_gt(:);
for i = 1:rows*cols
  [~,idx_b1] = min(abs(B1_gt(i)-MAPS(4,i,:)));
  maps(1,i) = MAPS(1,i,idx_b1);
  maps(2,i) = MAPS(2,i,idx_b1);
  maps(3,i) = MAPS(3,i,idx_b1);
  maps(4,i) = MAPS(4,i,idx_b1);
  maps(5,i) = MAPS(5,i,idx_b1);
end
t = toc;

fprintf('matching time=%.2f\n',t);

% rawrite(maps,path);
