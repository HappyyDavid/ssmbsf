function [Il_gray,Ir_gray,time_spent] = grayImages(Il,Ir)
% converts 4 views to grayscale images

tStart = tic;

[~,~,c] = size(Il);
if c==3
    Il_gray = rgb2gray(Il);
    Ir_gray = rgb2gray(Ir);
else
    Il_gray = Il;
    Ir_gray = Ir;
end
    
time_spent = toc(tStart);
% fprintf('\nrgb2gray computed in\t\t%6.2f s\n',time_spent);

end