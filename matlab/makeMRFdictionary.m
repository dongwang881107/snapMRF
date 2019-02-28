function dict=makeMRFdictionary(FA_v,phi_v,TR_v ,T1, T2, B0);
% This function calculates the MRF signal evolution for an IR-bSSFP-based readout % for a single material (T1, T2, B0) for the input RF pulse train and TR.
% This code is based on the single isochromat method popularized by Klaus
% Scheffler. This works for bSSFP as long as one can assume that both the
% flip angle and off-resonance frequency are essentially homogeneous inside % a voxel.
%
% %
% %
% % %
%
%
%

  N=length(TR_v);
  rf=FA_v*pi/180;
  rph=phi_v*pi/180;
  dict=zeros(1,N);
  m1=[0 0 -1].';    % This assumes a perfect inversion pulse with no delay
 for i=1:N
     rx=    [   1.0 0.0 0.0;        %rotation matrix for pulse
                0.0 cos(rf(i)) sin(rf(i));
                0.0 -sin(rf(i)) cos(rf(i))];
     rdzp=  [   cos(rph(i)) sin(rph(i)) 0.0; %RF phase
                -sin(rph(i)) cos(rph(i)) 0.0;
                0.0 0.0 1.0];
     rdzm=  [   cos(-rph(i)) sin(-rph(i)) 0.0;
                -sin(-rph(i)) cos(-rph(i)) 0.0;
                0.0 0.0 1.0];

     e1=exp(-TR_v(i)./T1);        % relaxation terms
     e2=exp(-TR_v(i)./T2);

     beta=B0.*TR_v(i)*2*3.141593/1000;

     rbeta= [ cos(beta./2) sin(beta./2) 0.0; % 1/2 off-resonance rotation
          -sin(beta./2) cos(beta./2) 0.0;
              0.0 0.0 1.0];
     %
     e12= [ e2 0.0 0.0;
            0.0 e2 0.0;
            0.0 0.0 e1];

     % m1=rx*rdzm*m1;    % do RF pulse
     m1=rdzp*rx*rdzm*m1;
     m1=(e12*m1+(1-e1)*[0 0 1].');  %relax
     m1=rbeta*m1;   % Off-resonance for first 1/2 of TR
     % dict(i)=m1(1)+j.*m1(2);  %Sample assuming TE=TR/2
     dict(i) = complex(m1(1),m1(2));
     m1=rbeta*m1;  % Off-resonance for 2nd 1/2 of TR
end
