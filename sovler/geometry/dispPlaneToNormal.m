function n = dispPlaneToNormal(dispPlane,K,base_m)
% computes 3D scaled normal from disparity plane

% Parameters
%   dispPlane   coefficients of the disparity plane in (a,b,c)-representation
%   K           3x3 camera matrix containing principal distance and point
%   base_m      (positive) length of the stereo baseline

    % camera parameters
    f  = K(1,1);
    cu = K(1,3);
    cv = K(2,3);
    
    % parameters of the disparity plane  
    % disp = a*u + b*v + c
    a = dispPlane(1);
    b = dispPlane(2);
    c = dispPlane(3);
        
    % transform disparity plane to scaled normal
    n    = zeros(3,1);
    n(1) = -a/base_m;
    n(2) = -b/base_m;
    n(3) = -(c + cu*a + cv*b) / (f*base_m);
    
end