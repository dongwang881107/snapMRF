function output = MRF_dict_B1(input)
% Extended Phase Graph signal generation of MRF signal from Jiang et al., MRM, 2015

%   INPUT: input.nreps = number of frames (or TRs) in MRF sequence
%              ".TI = inversion time
%              ".delk = pos integer is step between states equal to a full
%                   dephasing imparted by crusher gradient
%              ".szomega = pos integer is number of factors of k to
%                   include in phase history
%              ".T1_v = vector of T1s in dictionary
%              ".T2_v = vector of T2s in dictionary
%              ".B1_v = vector of B1 multipliers in dictionary
%              ".TR_v = vector of repetition times
%              ".TE_v = vector of echo times
%              ".FA_v = vector of flip angles in degrees
%              ".phi_v = vector of RF phases
%              ".reduce = 1 then do SVD compression of dictionary
%              ".sfrac = fraction of dictionary energy to retain if
%                   compressing

%   OUTPUT: output.dict_list = matrix describing T1, T2, B1, etc. for all
%                       realisitic combinations
%                ".dict_norm = normalized dictionary w/columns
%                       parameterized by dict_list
%                ".dict = dictionary w/columns parameterized by dict_list
%                ".V_red = reduced right singular vectors of dictionary
%                ".dict_red = compressed dictionary
%                ". = all input params copied to output params
%                ".B1_compress_dict_list_v = vector of B1s describing column
%                       B1s of compressed dictionary and associate matrices


disp('Constructing MRF dictionary...')
tic
%% declare parameters
nreps = input.nreps;
TI = input.TI;
delk = 1; % step between states equal to a full dephasing imparted by crusher gradient
szomega = 101; % number of factors of k to include in phase history
T1_v = input.T1_v;
T2_v = input.T2_v;
B1_v = input.B1_v;
TR_v = input.TR_v;
TE_v = input.TE_v;
FA_v = input.FA_v;
phi_v = input.phi_v;

nT2 = numel(T2_v);
nT1 = numel(T1_v);
nB1 = numel(B1_v);

%% dictionary init

dict_list = zeros(nT2*nT1*nB1,3);

% determine dictionary list
nn = 1;
for kk = 1:nB1

    B1 = B1_v(kk);

    for jj = 1:nT2

        T2 = T2_v(jj);

        for ii = 1:nT1

            T1 = T1_v(ii);
            if T1 > T2

                dict_list(nn,:) = [T1, T2, B1];
                nn = nn+1;

            end

        end
    end
end

% remove zero rows from listT1T2_m, normFt_m, Ft_m since init matrices are
% too big
dict_list(~any(dict_list,2),:) = [];

dict_length = size( dict_list,1 );

dict = zeros(nreps,dict_length);
dict_norm = dict;

%% dictionary generation

% my_pool = parpool([1 2],'SpmdEnabled',false);

T1_list = dict_list(:,1);
T2_list = dict_list(:,2);
% B0_list = dict_list(:,3);
B1_list = dict_list(:,3);

for ii = 1:dict_length

    T1 = T1_list(ii);
    T2 = T2_list(ii);
%     B0 = B0_list(ii);
    B1 = B1_list(ii);
    myFA_v = FA_v.*B1;

    disp(['MRF dictionary construction: T1 ',num2str(T1),' :: T2 ',...
        num2str(T2), ' :: B1 ', num2str(B1)]);

    % run sequence using EPG
    tmp_v = EPG_MRF_SSFP(T1,T2,0,TE_v,TR_v,myFA_v,delk,nreps,szomega,phi_v,TI );

    dict(:,ii) = tmp_v;
    dict_norm(:,ii) = tmp_v./sqrt(tmp_v*tmp_v');

end


% create output structure
output = input;

output.dict_list = dict_list;
output.dict_norm = dict_norm;
output.dict = dict;

%% determine reduced dictionary space

% reduce dimensionality of dictionary via SVD method by McGivney et al,
% IEEE MI, 2014

if input.reduce == 1
    disp('Compressing MRF dictionary by SVD...')

    output.U_r = [];
    output.dict_compress = [];
    output.B1_compress_dict_list_v = [];
    for ii = 1:numel(B1_v); % compress dictionary by B1 discretizations

        B1 = B1_v(ii);
        my_dict_norm = dict_norm(:, (dict_list(:,3) == B1) );

        [U,S,~] = svd(my_dict_norm,'econ');

        if ii == 1
            s_v = diag(S);
            fNRG_v = cumsum(s_v.^2)./sum(s_v.^2);
            nDictSpace = sum(fNRG_v <= input.sfrac);
        end

        U_r = U(:,1:nDictSpace);
        my_dict_compress = U_r'*my_dict_norm;

        output.U_r = [output.U_r U_r];
        output.dict_compress = [output.dict_compress my_dict_compress];
        output.B1_compress_dict_list_v = [output.B1_compress_dict_list_v B1*ones(1,nDictSpace)];

    end

    disp('Compression of MRF dictionary by SVD complete.')
end


%%

% delete(my_pool);

t = toc;
disp(['Construction of MRF dictionary complete. Elapsed time is ' num2str(t) ' s.'])

end
