function Ft_v = EPG_MRF_SSFP( T1,T2,B0,TE_v,TR_v,FA_v,delk,nreps,szomega,phi_v,TI)
% returns the complex transverse magnetization (Mx + iMy)
% using the extended phase graph approach for a magnetic resonance
% fingeprinting unbalanced SSFP sequence similar to that used in Jiang et al, MRM,
% 2015

%   INPUT: T2 = T2 of the imaged object (T2* is neglected)
%          TR_v = vector of repetition times
%          FA_v = vector of flip angles in degrees
%          delk = pos integer step of a full dephasing by the crusher
%          szomega = pos integer length of the state vectors used to
%               calculate the magnetization by the EPG formalism (too short
%               a vector will change the answer since some
%               magnetization with be lost by a truncated state vector)
%          TE_v = vector of echo times

%   OUTPUT: Ft_v = vector of length nreps containting the complex trans
%                   mag


% convert phi_v to radians
phi_v = phi_v.*pi/180;
% i_sign = 1;
% initialize magnetization w/inversion
omega = [[0,0,-1]' zeros(3,szomega-1)];

% inversion delay
% TI = 0; % ms
e1 = exp(-TI/T1);
E = diag([0 0 e1]);
omega(:,1) = E*omega(:,1) + [0 0 (1-e1)]';

% iterate for defined reps
Ft_v = zeros(1,nreps);
for j=1:nreps

    alpha = FA_v(j);
    TR = TR_v(j);
    phi = phi_v(j);
    TE = TE_v(j);

    % flip angle transition
    T_m = rfTransEPG(alpha,phi);
    omega = T_m*omega;

    % decay for readout at TE
    e2 = exp(-TE/T2);
    e1 = exp(-TE/T1);
    E = diag([e2 e2 e1]);
    omega(:,2:end) = E*omega(:,2:end);
    omega(:,1) = E*omega(:,1) + [0 0 (1-e1)]';

    % shift phase for B0
    omega(1,:) = omega(1,:).* exp( 1i * 2 * 3.141593 * B0 * TE /1000);
    omega(2,:) = omega(2,:).* exp( -1i * 2* 3.141593 * B0 * TE /1000);

    % omega(1,:) = omega(1,:).* exp( 1i * 2 *3.141593 * TE );
    % omega(2,:) = omega(2,:).* exp( -1i * 2*3.141593  * TE );

    % omega(1,:) = omega(1,:).*exp(1i*B0);
    % omega(2,:) = omega(2,:).*exp(1i*B0);

    % store readout transverse magnetization
    Ft_v(j) = omega(1,1);

    % shift phase for B0
    omega(1,:) = omega(1,:).* exp(  1i * 2 * 3.141593 * B0 * (TR-TE) /1000);
    omega(2,:) = omega(2,:).* exp(  -1i * 2 * 3.141593  * B0 * (TR-TE) /1000);

    % omega(1,:) = omega(1,:).* exp(  1i * 2* 3.141593* (TR-TE) );
    % omega(2,:) = omega(2,:).* exp(  -1i * 2* 3.141593* (TR-TE) );

    % omega(1,:) = omega(1,:).*exp(1i*B0);
    % omega(2,:) = omega(2,:).*exp(1i*B0);

    % apply gradient dephasing specified by delk
    omega(1,:) = circshift(omega(1,:),delk,2);
    omega(2,:) = circshift(omega(2,:),-delk,2);
    omega(1,1:delk) = 0;
    omega(2,(end-delk+1):end) = 0;
    omega(1,1) = conj(omega(2,1));

    % decay for flip angle at TR
    e2 = exp(-(TR-TE)/T2);
    e1 = exp(-(TR-TE)/T1);
    E = diag([e2 e2 e1]);
    omega(:,2:end) = E*omega(:,2:end);
    omega(:,1) = E*omega(:,1) + [0 0 (1-e1)]';

end

end
