% SuperpixelBoundaries - Compute the boundary pixels of every superpixel
%
% sup_boundaries = SuperpixelBoundaries(seg_image, boundary)
% For every superpixel in seg_image, find the boundaries in boundary which
% are the boundaries of that superpixel. Return a list of linear boundary
% pixel indices for each superpixel.
%
% Arguments:
%   seg_image - MxN superpixel segmentation
%  
% Returns:
%   sup_boundaries - num_sups x 1 cell array of superpixel boundaries. Each
%   entry contains a list of linear pixel indices of the boundary pixels
%   for the given superpixel.
%
function sup_boundaries = superpixel_boundaries(M,seg_image)

[h,w] = size(seg_image);
num_sups = max(seg_image(:));
sup_boundaries = cell(num_sups, 1);

for sup = 1:num_sups
  BW = M{sup};
  [v,u] = find(BW);
  u1 = min(u);
  v1 = min(v);
  u2 = max(u);
  v2 = max(v);
  BW = BW(v1:v2,u1:u2);
  b = bwboundaries(BW,8,'noholes'); % Trace region boundaries in binary image
  b = b{1};
  % like filtering index
  d = [b; ...
       b+ones(size(b,1),1)*[-1 0]; ...
       b+ones(size(b,1),1)*[+1 0]; ...
       b+ones(size(b,1),1)*[0 -1]; ...
       b+ones(size(b,1),1)*[0 +1]];
  d = d+ones(size(d,1),1)*[v1-1 u1-1];
  d(:,1) = min(max(d(:,1),1),h);
  d(:,2) = min(max(d(:,2),1),w);

  % compute sorted indices (sort required by pairwise energy mex wrapper)
  idx = sub2ind([h,w],d(:,1),d(:,2));
  idx = unique(idx); % return valuable is in sorted order
  sup_boundaries{sup} = idx;
end
