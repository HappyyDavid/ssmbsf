function C = superpixelCenters(S,Pd)
% computes 3D coordinates for superpixel centroids from current disp planes

  uvd = [ [S.cu]',[S.cv]',-ones(numel(S),1)];

  for iS = 1:numel(S)
    
    if S(iS).gamma < 0
      continue;
    end
    
    uvd(iS,3) = S(iS).gamma;
    
  end
  
  C = project(uvd,inv(Pd));
  
end