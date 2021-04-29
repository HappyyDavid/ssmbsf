function dispPlane = normalToDispPlane(normal,K,base_m)
% computes disparity plane from 3D scaled normal

% Parameters
%   normal      normal of the 3D plane divided by distance to origin
%   K           3x3 camera matrix containing principal distance and point
%   base_m      (positive) length of the stereo baseline

    % camera parameters
    f  = K(1,1);
    cu = K(1,3);
    cv = K(2,3);

    % disparity plane parameters
    a = -base_m * normal(1);
    b = -base_m * normal(2);
    c = cu*base_m*normal(1)+cv*base_m*normal(2)-f*base_m*normal(3);
    
    % disparity plane conventions:
    % disp = alpha*u + beta*v + gamma
    dispPlane = [a,b,c];
    
end