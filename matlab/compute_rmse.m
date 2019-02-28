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
t1_gt = [2480,2173,1907,1604,1332,1044,801.7,608.6,458.4,336.5,244.2,176.6,126.9,90.0];
t1_gt = sort(t1_gt');
t2_gt = [581.3,403.5,278.1,190.9,133.3,96.9,64.1,46.4,32,22.6,15.8,11.2,8,5.6];
t2_gt = sort(t2_gt');

% t1_acc = sqrt(sum(norm(t1_gt-t1_ave))/14)/(max(t1_ave)-min(t1_ave))
% t2_acc = sqrt(sum(norm(t2_gt-t2_ave))/14)/(max(t2_ave)-min(t2_ave))
t1_acc_epg = sqrt(sum(norm(t1_gt-t1_ave_epg))/14);
t2_acc_epg = sqrt(sum(norm(t2_gt-t2_ave_epg))/14);
t1_acc_roa = sqrt(sum(norm(t1_gt-t1_ave_roa))/14);
t2_acc_roa = sqrt(sum(norm(t2_gt-t2_ave_roa))/14);
%%

t = [t1_gt,t1_ave_epg,t1_ave_roa,t2_gt,t2_ave_epg,t2_ave_roa];
% T = cell(16,4);

%% Save into .csv file
colnames = {'a','b','c','d','e','f'};

t=str2num(num2str(t,3));
t = num2cell(t);

T = cell(15,6);
T(1:14,:) = t;
T(15,1) = {"RMSE (ms)"};
T(15,2) = num2cell(str2num(num2str(t1_acc_epg,3)));
T(15,3) = num2cell(str2num(num2str(t1_acc_roa,3)));
T(15,4) = {"RMSE (ms)"};
T(15,5) = num2cell(str2num(num2str(t2_acc_epg,3)));
T(15,6) = num2cell(str2num(num2str(t2_acc_roa,3)));

%% Write results to csv files
% data = table(t(:,1),t(:,2),t(:,3),t(:,4),t(:,5),t(:,6),'VariableNames',colnames);
% writetable(data,'/Users/wangdong/Documents/Literature/My Papers/mrf/table1.csv');

% data = table(T(end,1),T(end,2),T(end,3),T(end,4),T(end,5),T(end,6),'VariableNames',colnames);
% writetable(data,'/Users/wangdong/Documents/Literature/My Papers/mrf/table2.csv');

%% Run these two lines in the shell to generate tex files
% cat table2.csv | sed 's/[ ,][ ,]*/\&/g' | awk '{print $0,"\\\\"}' | tail -1 > table2.tex
% cat table1.csv | sed 's/,/\&/g' | awk '{print $0,"\\\\"}' > table1.tex
