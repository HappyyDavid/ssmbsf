function alphaBetaGamma = alphaBetaGammaFromAbc( planes_abc,centers )
%ALPHABETAGAMMAFROMABC conversion of disparity plane representation

    alphaBetaGamma = planes_abc;

    for iP = 1:size(centers,1)
        a = planes_abc(iP,1,:);
        b = planes_abc(iP,2,:);
        c = planes_abc(iP,3,:);
        
        alphaBetaGamma(iP,3,:) = c + a * centers(iP,1) + b * centers(iP,2); 
    end

end

