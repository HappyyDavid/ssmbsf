function [Tr,R,t] = getRigidMotion( X1,X2 )
%GETRIGIDMOTION 

    % estimate rigid motion between point clouds
    % X2 = Tr*X1
    % cf. Arun, PAMI 87: Least-Squares Fitting of Two 3-D Point Sets

    % Step 1) centroids and centered coordinates
    X1_mean = mean(X1,1);
    X2_mean = mean(X2,1);

    X1_c = bsxfun(@minus,X1,X1_mean);
    X2_c = bsxfun(@minus,X2,X2_mean);

    % Step 2) compute H
    H = X1_c'*X2_c;

    % Step 3) SVD of H
    [U,~,V] = svd(H);

    % Step 4) compute R
    R = V*U';

    % Compute T
    t = X2_mean'-R*X1_mean';
    
    % Transformation matrix
    Tr = [R,t;0,0,0,1];

end

