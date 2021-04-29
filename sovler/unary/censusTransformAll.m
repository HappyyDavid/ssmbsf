function [ Cl,Cr,time_spent ] = censusTransformAll( Il,Ir )
%CENSUSTRANSFORMALL Computes census transformed images for four views

tStart = tic;

[~,~,ch] = size(Il);
assert(ch==1,'input images must be single-channel');

Cl = colfilt(Il,[5 5],'sliding',@census); % columnwise neighborhood operations
Cr = colfilt(Ir,[5 5],'sliding',@census);


time_spent = toc(tStart);
% fprintf('\nCensus transform computed in\t%6.2f s\n',time_spent);

end

