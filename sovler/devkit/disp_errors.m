function [errs, d_err] = disp_errors(D_gt,D_est,tau)
    
    mask  = find(D_gt>0);
    D_gt  = D_gt(mask);
    D_est = D_est(mask);
    
    
    abs_rel  = mean(abs(D_gt-D_est)./D_gt);
    rmse     = sqrt(mean((D_gt-D_est).^2));
    rmse_log = sqrt(mean((log(D_gt)-log(D_est)).^2));
    sq_rel   = mean((D_gt-D_est).^2./D_gt);
    
    thresh = max((D_gt./D_est),(D_est./D_gt));
    a1 = mean((thresh<1.25));
    a2 = mean((thresh<1.25^2));
    a3 = mean((thresh<1.25^3));
    
    E = abs(D_gt-D_est);
    n_err   = length(find(D_gt>0 & E>tau(1) & E./abs(D_gt)>tau(2)));
    n_total = length(find(D_gt>0));
    d_err = n_err/n_total;
    
    errs = [abs_rel, rmse, rmse_log, sq_rel, a1, a2, a3];