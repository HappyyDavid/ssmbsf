function [ factors_pw, time_spent ] = computeSmoothnessTerm( ...
           S,adjMat,sup_boundaries,shapeParticles,h,K,b_m,sNorm,sDept,wNormalSmooth,wDispSmooth)
% construct plane orient smooth, shape smooth error funtion
    tStart = tic; % time
    [nS, ~, nP] = size(shapeParticles);
    factors_pw = cell(1, 1);
    
    abcShapeParticles = abcFromAlphaBetaGamma(shapeParticles, [S.cu;S.cv]');
    
    factor_idx = 1;
    for sp_idx = 1:nS % loop all superpixels
        
        for sp_jdx = sp_idx+1:nS % loop all remains superpixels
            
            % check wether superpixel i, j is adjacent, 1 means they are adjacence.
            if adjMat(sp_idx, sp_jdx) > 0.5
                
                pair_error = zeros(nP, nP); % save pairwise term in matrix
                % get pixels of common boundary between adjacent superpixels
                boundaryIndex = intersect(sup_boundaries{sp_idx}, sup_boundaries{sp_jdx});        
                u = floor(boundaryIndex/h) +1; % column index
                v = mod(boundaryIndex, h) +1; % raw index
                        
                for p_idx = 1:nP % loop first shapearticles 
                    
                    for p_jdx = 1:nP % loop second shapeparticles
                        % shape smooth term
                        
                        alpha_1  = shapeParticles(sp_idx, 1, p_idx); % disparity plane alpha, beta, gamma
                         beta_1  = shapeParticles(sp_idx, 2, p_idx);
                        gamma_1  = shapeParticles(sp_idx, 3, p_idx);
                        
                        alpha_2  = shapeParticles(sp_jdx, 1, p_jdx); 
                         beta_2  = shapeParticles(sp_jdx, 2, p_jdx);
                        gamma_2  = shapeParticles(sp_jdx, 3, p_jdx);
                        
                        D1 = gamma_1 + alpha_1*(u - S(sp_idx).cu) + beta_1*(v - S(sp_idx).cv);
                        D2 = gamma_2 + alpha_2*(u - S(sp_jdx).cu) + beta_2*(v - S(sp_jdx).cv);
                        
                        dDept = abs(D1-D2);
                        dDept(dDept>sDept) = sDept; % truncated L1 
                        disp_error = sum(dDept);
                
                        % orient smooth term
                        n_i = abcShapeParticles(sp_idx, 1:3, p_idx); % disparity plane abc
                        n_j = abcShapeParticles(sp_jdx, 1:3, p_jdx); 
                        n_i = dispPlaneToNormal(n_i, K, b_m); % normal from disparity plane abc
                        n_j = dispPlaneToNormal(n_j, K, b_m);
                        
                        cos_sim = (n_i' * n_j) / (norm(n_i)*norm(n_j));
                        dNorm = 0.5*(1.0-cos_sim);
                        norm_error = min(dNorm, sNorm); % truncated L1 
                        
                        pair_error(p_idx, p_jdx) = wDispSmooth*disp_error + wNormalSmooth*norm_error; % is right order?
                    end
                end
                % save factors
                factors_pw{factor_idx}.v = [sp_idx, sp_jdx];
                factors_pw{factor_idx}.e = pair_error(:);
                factor_idx = factor_idx + 1;
            end
            
        end
    end
    
    time_spent = toc(tStart);
end

