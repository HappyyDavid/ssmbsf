% Arguments:
%   seg_image - superpixel segmentation. MxN matrix.
%
% Returns:
%   adjmat - num_sups x num_sups strictly upper triangular adjacency matrix.
%   adjmat(i,j) = 1 iff superpixels i and j are adjacent (0 on diagonal)

function adjmat = superpixel_adjacencies(seg_image)

num_sups = max(seg_image(:));
adjmat = zeros(num_sups, num_sups);
[M,N] = size(seg_image);

for x = 1:M-1
  for y = 1:N-1
    s  = seg_image(x,   y  );
    s1 = seg_image(x+1, y  );
    s2 = seg_image(x,   y+1);
    s3 = seg_image(x+1, y+1);
    adjmat(s, s1) = 1;
    adjmat(s1, s) = 1;
    adjmat(s, s2) = 1;
    adjmat(s2, s) = 1;
    adjmat(s, s3) = 1;
    adjmat(s3, s) = 1;
  end
end

% set all lower triangular values (including diagonal) to zero
adjmat = adjmat.*triu(ones(num_sups,num_sups),1);
