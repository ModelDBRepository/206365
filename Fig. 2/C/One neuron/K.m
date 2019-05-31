function K = K(a,b)

% normalization for second order approximation

if a~=b
K=a*b/(a-b)*((a/b)^(b/(b-a))-(a/b)^(a/(b-a)));

else
K=a*exp(-1);

end;

end

