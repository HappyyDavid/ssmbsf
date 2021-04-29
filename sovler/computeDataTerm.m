function [factors_data, time_spent] = computeDataTerm(S,shapeParticles,D,wData,Cl,Cr,Il,Ir,h,w)
% compute data term by per shape particle stored in n_superpixel * n_particles
    tStart = tic;
    threshCensus = 0.8361;
    occCensus = 0.3590;
    
    [nS, ~, nP] = size(shapeParticles);
    
    % init unary factors
    factors_data = cell(1, nS);
    for iS = 1:nS
        factors_data{iS}.v = iS;
        factors_data{iS}.e = zeros(1, nP);
    end
    
    for iS = 1:nS % loop all superpixels
            
            D_ori = D(S(iS).idx);
            
            for iP = 1:nP % loop all shape particles
                E_cens = 0;
                
                alpha = shapeParticles(iS, 1, iP);
                beta  = shapeParticles(iS, 2, iP);
                gamma = shapeParticles(iS, 3, iP); % i think use original data is better
                D_opt = gamma + alpha * (S(iS).u - S(iS).cu) + beta * (S(iS).v - S(iS).cv);
                
                % compute data term in image space
                     
                v2 = S(iS).v; % 
                u2 = S(iS).u-round(D_opt); % TODO CHECK how to use disparity
                outer_mask = (u2<1)|(u2>w);
                
                u2(u2<1)=1; u2(u2>w)=w;
                idx2 = sub2ind([h,w],v2,u2);
                cl = Cl(S(iS).idx);
                cr = Cr(idx2);
                
                ham_dis = sum(dec2bin(bitxor(cl,cr,'uint32'))'-'0');
                tru_ham_dis = min(double(ham_dis)/24.0,threshCensus);
                %tru_ham_dis(outer_mask) = occCensus; % mask outlier
                
                %tru_ham_dis = tru_ham_dis(unigrid(1,4,S(iS).nPix));
                      
                E_cens = sum(tru_ham_dis);    

                error = sum(abs(D_ori-D_opt));
                factors_data{iS}.e(iP) = wData*error + E_cens;
            end
        
    end
   
    time_spent = toc(tStart); 

end

% census + hamming distance, very slow
%{
                % compute data term in image space
                for iB = 1:S(iS).nPix % loop all pixels
                    disp(iB);
                    v2 = S(iS).v(iB);
                    u2 = S(iS).u(iB)+round(D_opt(iB));
                    if (u2 < 1) || (u2 > w)  % check correspondence pixel inside right image
                        E_cens = E_cens + occCensus; % TODO 
                    else
                        cl = Cl(S(iS).idx(iB));
                        cr = Cr(v2, u2);
                        ham_dis = sum(de2bi(bitxor(cl,cr))); % xor in bit format, convert result to binary, sum of all elements.
                        E_cens = E_cens + min(double(ham_dis)/24.0, threshCensus);
                    end
                end

                v2 = S(iS).v; % TODO use RGB image
                u2_ori = S(iS).u+round(D_opt);
                u2 = u2_ori;
                u2(u2<1)=1; u2(u2>w)=w;
                idx2 = sub2ind([h,w],v2,u2);
                gray1 = Il(S(iS).idx);
                gray2 = Ir(idx2);
                diff_gray = abs(int32(gray1)-int32(gray2));
                E_cens = sum(diff_gray);  
%}

