function [flux_d,soln]=EllipticSolver(d, Nx, sigma)
% d is a vector of size 4*Nx that tabulates the Dirichlet data on the boundary
% d is oriented counter-clockwisely, see dirichlet_index in the code for
% more details
% flux_d computes the flux of solution on the boundary, same size as d
% soln is the solution to the PDE, of size (Nx+1) by (Nx+1)

dx = 1/Nx;
x0 = 0:dx:1;
y0 = x0;
[xx,yy] = meshgrid(x0,y0);
% [X,Y] = meshgrid(1/(2*Nx):1/Nx:(2*Nx-1)/(2*Nx),1/(2*Nx):1/Nx:(2*Nx-1)/(2*Nx));
[X, Y] = meshgrid(linspace(0, 1, Nx), linspace(0, 1, Nx));

%% build coefficient matrices
% sigma_fun = @(x,y)(1+x-x+y-y);      % setting media as a function of x and y here

StiffA = zeros((Nx+1)^2,(Nx+1)^2);
soln = zeros(Nx+1,Nx+1); soln = soln(:); 

stiff_local_l = [1,-1/2,0,-1/2;-1/2,1/2,0,0;0,0,0,0;-1/2,0,0,1/2];
stiff_local_r = [0,0,0,0;0,1/2,-1/2,0;0,-1/2,1,-1/2;0,0,-1/2,1/2];

% do interpolation below to get the sigma_l and sigma_r
sigma_l = interp2(xx, yy, sigma, xx(1:end-1,1:end-1) + dx/3, yy(1:end-1,1:end-1) + 2*dx/3);
sigma_r = interp2(xx, yy, sigma, xx(1:end-1,1:end-1) + 2*dx/3, yy(1:end-1,1:end-1) + dx/3);

% sigma_l = sigma_fun(xx(1:end-1,1:end-1) + dx/3,yy(1:end-1,1:end-1) + 2*dx/3); 
% sigma_r = sigma_fun(xx(1:end-1,1:end-1) + 2*dx/3,yy(1:end-1,1:end-1) + dx/3);

for km = 1:Nx % x direction coordinate
    for kn = 1:Nx % y direction coordinate
        indices = index_map(km,kn,Nx);
        StiffA(indices,indices) = StiffA(indices,indices) + sigma_l(km,kn)*stiff_local_l;
        StiffA(indices,indices) = StiffA(indices,indices) + sigma_r(km,kn)*stiff_local_r;
    end
end

%% build index for the boundary and interior
index_matrix = 1:(Nx+1)^2; 
index_matrix = reshape(index_matrix,Nx+1,Nx+1);
dirichlet_index = [index_matrix(1:end-1,1);index_matrix(end,1:end-1)';...
     index_matrix(end:-1:2,end);index_matrix(1,end:-1:2)']; % index clockwise starting

unknown_index = index_matrix(2:end-1,2:end-1); unknown_index = unknown_index(:);

%% solving PDE

soln(dirichlet_index) = d;  % load boundary condition
load = -StiffA(:,dirichlet_index)*soln(dirichlet_index); load = load(unknown_index);
% figure(3);mesh(reshape(load,Nx,Nx));pause;

soln(unknown_index)= StiffA(unknown_index,unknown_index)\load;

flux = StiffA*soln/dx;
flux_d = flux(dirichlet_index);
soln = reshape(soln,Nx+1,Nx+1);
end


function indices = index_map(km,kn,Nx_H)
    indices(1) = (km-1)*(Nx_H+1)+kn;
    indices(2) = (km-1)*(Nx_H+1)+kn+1;
    indices(3) = (km)*(Nx_H+1)+kn+1;
    indices(4) = (km)*(Nx_H+1)+kn;
end
