function D = dispMapFromPlanes( S,planes,h,w )
%DISPMAPFROMPLANES disparity map from superpixels and
% (alpha,beta,gamma)-planes

  nS = numel(S);
  D  = zeros(h,w);

  for iS=1:nS
      nP = numel(S(iS).idx);
      D( S(iS).idx ) = [S(iS).u-S(iS).cu,S(iS).v-S(iS).cv,ones(nP,1)]*planes(iS,:)';
  end

end

