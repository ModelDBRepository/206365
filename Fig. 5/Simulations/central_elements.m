function central_elements = central_elements(n,p)
%UNTITLED4 Summary of this function goes here

%%

p=p-1;             % to have proper number of elements
h=0.5*(n-p);       % half length in the matrix
A=zeros(n,n);
k=0;

%   calculates the indexes for central elements
for i=h:1:h+p
    for j=h:1:h+p     
        k=k+1;
        A(i,j)=1;      
        IND(k)=j+(i-1)*n;
    end
end

central_elements=IND;

%%

end