clear

maps_epg = raread('../result/maps400k_phantom_varTR_b1map.ra');
maps_roa = raread('../result/maps400k_phantom_varTR_b1map_matlab.ra');

t1_epg = reshape(maps_epg(1,:),240,240);
t2_epg = reshape(maps_epg(2,:),240,240);
t1_roa = reshape(maps_roa(1,:),240,240);
t2_roa = reshape(maps_roa(2,:),240,240);

m = 240;
n = 240;
[cols,rows] = meshgrid(1:m,1:n);

r = 3;
x = [111,84,71,77,100,132,158,171,141,165,97,105,136,145];
y = [67,82,110,142,162,165,150,122,70,90,100,139,93,131];

t1_ave_epg = zeros(14,1);
t2_ave_epg = zeros(14,1);
t1_ave_roa = zeros(14,1);
t2_ave_roa = zeros(14,1);

circle = zeros(240,240);

for i = 1:14
    dist = (rows - y(i)).^2 + (cols - x(i)).^2;
    cir = dist <= r.^2;
    num = numel(find(cir~=0));
    t1_mask_epg = t1_epg.*cir;
    t1_mask_roa = t1_roa.*cir;
    t1_ave_epg(i,1) = sum(t1_mask_epg(:))/num;
    t1_ave_roa(i,1) = sum(t1_mask_roa(:))/num;
    t2_mask_epg = t2_epg.*cir;
    t2_mask_roa = t2_roa.*cir;
    t2_ave_epg(i,1) = sum(t2_mask_epg(:))/num;
    t2_ave_roa(i,1) = sum(t2_mask_roa(:))/num;

    circle=circle+cir;
end

t1_ave_epg = sort(t1_ave_epg);
t2_ave_epg = sort(t2_ave_epg);
t1_ave_roa = sort(t1_ave_roa);
t2_ave_roa = sort(t2_ave_roa);

%%
t1_gt = [2480,2173,1907,1604,1332,1044,801.7,608.6,458.4,336.5,244.2,176.6,126.9,90.9];
t1_gt = sort(t1_gt');
t2_gt = [581.3,403.5,278.1,190.94,133.27,96.89,64.07,46.42,31.97,22.56,15.813,11.237,7.911,5.592];
t2_gt = sort(t2_gt');

t1_acc_epg = sqrt(sum(norm(t1_gt-t1_ave_epg))/14);
t2_acc_epg = sqrt(sum(norm(t2_gt-t2_ave_epg))/14);
t1_acc_roa = sqrt(sum(norm(t1_gt-t1_ave_roa))/14);
t2_acc_roa = sqrt(sum(norm(t2_gt-t2_ave_roa))/14);
%%
t = [t1_gt,t1_ave_epg,t1_ave_roa,t2_gt,t2_ave_epg,t2_ave_roa];

f = fopen('table1.tex','w');

for i=1:size(t,1)
    for j=1:size(t,2)
        if (j<size(t,2))
            fprintf(f,'%.1f & ', t(i,j));
%             fprintf(f, '%.*f & ', 2-floor(log10(abs(t(i,j)))), t(i,j));
        else
            fprintf(f,  '%.1f\\\\\n', t(i,j));
%             fprintf(f, '%.*f\\\\\n', 2-floor(log10(abs(t(i,j)))), t(i,j)); 
        end
    end
end
fprintf(f,'\\midrule[1pt]\n');
fprintf(f, 'RMSE (ms) & %.1f & %.1f & RMSE (ms) & %.1f & %.1f\\\\\n', ...
    t1_acc_epg, t1_acc_roa, t2_acc_epg, t2_acc_roa);

fclose(f);

