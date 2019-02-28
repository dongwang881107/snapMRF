%% Generate binary mask

clear;

maps = raread('../result/maps400k_phantom_varTR_b1map.ra');
pd = reshape(maps(5,:),[240,240]);

%%
mask = (pd>0.049);
mask = bwconvhull(mask);
mask = single(mask);
imshow(mask);
