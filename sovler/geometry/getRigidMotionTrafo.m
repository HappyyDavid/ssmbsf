function Tr = getRigidMotionTrafo(rx_deg_,ry_deg_,rz_deg_,tx,ty,tz)

    Rx = rx_deg(rx_deg_);
    Ry = ry_deg(ry_deg_);
    Rz = rz_deg(rz_deg_);

    R  = Rx * Ry * Rz; 
    
    Tr = [R,[tx;ty;tz];0,0,0,1];

end