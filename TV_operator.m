%% TV_operator
%
% Description: 
%  Function to compute the one-dimensional TV operator 
% 
% INPUT: 
%  n :    	number of equidistant grid points 
%  order :	order of the TV operator 
%
% OUTPUT: 
%  R :    	matrix representation of the TV operator 
%
% Author: Jan Glaubitz 
% Date: Jan 07, 2022  
%

function R = TV_operator( n, order )

    e = ones(n,1);
    
    if order == 1
        D = spdiags([e -e], 0:1, n,n);
 	elseif order == 2
       	D = spdiags([-e 2*e -e], 0:2, n,n);
 	elseif order == 3
      	D = spdiags([e -3*e 3*e -e], 0:3, n,n); 
    else 
        error('Desried order not yet implemented!')
    end
    
    R = D(1:n-order,:);
        
end