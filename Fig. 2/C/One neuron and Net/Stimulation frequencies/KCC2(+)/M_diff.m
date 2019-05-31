function A = M_diff(N)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% creates the diffusion matrix of a size N
% lattice with periodic border conditions

function ind = graph_element(x,y,N)
%UNTITLED2 Summary of this function goes here
%   returns the number of an element on the graph, numbrering -> and down
ind=x+(y-1)*N;
end

% N - size parameter, number of elements in the matrix - sqrt(elements)

% Diffusion matrix

A=zeros(N^2,N^2); % adjacenty matrix, SYMMETRIC!

for x=1:1:N         % loop over all all graph edges
    for y=1:1:N
        
    % border conditions
        A(graph_element(x,1,N),graph_element(x,N,N))=1;        
        A(graph_element(1,y,N),graph_element(N,y,N))=1;
        
        A(graph_element(x,N,N),graph_element(x,1,N))=1;        
        A(graph_element(N,y,N),graph_element(1,y,N))=1;
        
       % formula for the rest of the elements
       if x+1<=N
      A(graph_element(x,y,N),graph_element(x+1,y,N))=1;                   
      A(graph_element(x+1,y,N),graph_element(x,y,N))=1;                     
      end
      
       if y+1<=N
       A(graph_element(x,y,N),graph_element(x,y+1,N))=1;
       A(graph_element(x,y+1,N),graph_element(x,y,N))=1;              
       end
       
       if x>1
       A(graph_element(x,y,N),graph_element(x-1,y,N))=1;            
       A(graph_element(x-1,y,N),graph_element(x,y,N))=1;              
       end       
    
       if y>1      
        A(graph_element(x,y,N),graph_element(x,y-1,N))=1;
        A(graph_element(x,y-1,N),graph_element(x,y,N))=1; 
       end                                
              
    end
end

end

