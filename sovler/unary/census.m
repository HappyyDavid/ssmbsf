function c = census(I)

    nP = size(I,1);          % number of pixels

    iM = ceil(nP/2);         % index of center element
    
    m  = I(iM,:);            % center element
    
    I(iM,:) = [];            % remove center element
     
    lbp = bsxfun(@lt,I,m);   % get local binary pattern
    
    T = 2 .^ (0:nP-2);       
    
    c = uint32(T*lbp);

end