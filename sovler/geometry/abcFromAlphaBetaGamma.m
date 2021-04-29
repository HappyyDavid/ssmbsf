function planes_abc = abcFromAlphaBetaGamma( planes,centers )
%ABCFROMALPHABETAGAMMA converts centered disparity plane representation to a,b,c

% planes is expected to be a n x 3 x m matrix where the
%   first index refers to a plane
%   second index references alpha, beta, gamma (in this order)
%   third index corresponds to the number of instantiations of the n-th plane

% centers contains (u,v)-coordinates of the centroids alpha,beta,gamma refer to

planes_abc = planes;

nS = size(centers,1);

    for iS = 1:nS

        alpha   = planes(iS,1,:);
        beta    = planes(iS,2,:);
        gamma   = planes(iS,3,:);

        planes_abc(iS,3,:) = gamma - alpha * centers(iS,1) - beta * centers(iS,2); 

    end

end

