function ham_dis = hamming_distance( a,b )
% xor in bit format, convert result to binary, sum of all elements.
    ham_dis = sum(de2bi(bitxor(a,b)));
end

