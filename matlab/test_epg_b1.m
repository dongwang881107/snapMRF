%% run example MRF EPG uSSFP generation
clear;

%% generate sequence params
csvdat = csvread('../data/MRF_100.csv',1,0);
img = raread('../data/phantom_100.ra');
B1_gt = raread('../data/B1_phantom_100.ra');

nreps = size(csvdat, 1);
nominal_flip = 60;
TR_base = 16;
TE_base = 3.5;
TI = 40;

FA_v = nominal_flip*csvdat(:,1);
phi_v = csvdat(:,2);
TR_v = TR_base + csvdat(:,3);
TE_v = TE_base + csvdat(:,4);

delk = 1;
szomega = 101;

%% generate test signal

T1_v = unique([10:10:100]);
T2_v = unique([4:2:20]);
B0_v = unique([0]);
B1_v = unique([0.9:0.1:1.1]);

input.T1_v = T1_v;
input.T2_v = T2_v;
input.B0_v = B0_v;
input.B1_v = B1_v;
input.FA_v = FA_v;
input.phi_v = phi_v;
input.TR_v = TR_v;
input.TE_v = TE_v;

input.nreps = nreps;
input.TI = TI;
input.reduce = 0;

output = MRF_dict_B1(input);

%% matching
atoms = output.dict_norm;
[~,natoms] = size(atoms);
natoms_b1 = natoms/numel(B1_v);

params = zeros(4, natoms);
index = 0;
for r = 1:numel(B1_v)
  for k = 1:numel(B0_v)
    for j = 1:numel(T2_v)
      for i = 1:numel(T1_v)
        if T1_v(i) > T2_v(j)
          index = index + 1;
          params(1,index) = T1_v(i);
          params(2,index) = T2_v(j);
          params(3,index) = B0_v(k);
          params(4,index) = B1_v(r);
        end
      end
    end
  end
end

% img = raread('../data/img_100.ra');
% B1_gt = raread('../data/B1_phantom_100.ra');

[nreps,rows,cols] = size(img);
img = reshape(img,[nreps,rows*cols]);

maps = zeros(5,rows*cols);
MAPS = zeros(5,rows*cols,numel(B1_v));

tic;
for k = 1:numel(B1_v)
  range = (k-1)*natoms_b1+1:k*natoms_b1;
  [proton, idx] = max(abs(atoms(:,range)'*img));
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
  [~,idx_b1]=min(B1_gt(i)-MAPS(4,i,:));
  maps(1,i) = MAPS(1,i,idx_b1);
  maps(2,i) = MAPS(2,i,idx_b1);
  maps(3,i) = MAPS(3,i,idx_b1);
  maps(4,i) = MAPS(4,i,idx_b1);
  maps(5,i) = MAPS(5,i,idx_b1);
end
t = toc;

fprintf("matching time=%.2f\n",t);

rawrite(maps,"../result/maps_matlab_epg_b1.ra");
