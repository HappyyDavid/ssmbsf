function [M,m] = superpixel_masks(SP)
% extract superpixel masks

num_superpx = double(max(SP(:)));

M = cell(num_superpx,1);
m = cell(num_superpx,1);
for i=1:num_superpx
  M{i} = SP==i;
  m{i} = find(M{i});
end
