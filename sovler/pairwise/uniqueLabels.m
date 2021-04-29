function [ L ] = uniqueLabels( L )

  % get unique labels
  uL = unique(L(:));
  
  % get number of unique labels
  nL = double(numel(uL));
  
  l=zeros(size(L));
  
  for i=1:nL
    l(L==uL(i))=i;
  end
  
  L=l;
  
end

