%% setup scaling parameters
clear
addpath('src')
N = 80;
h =  1/(N-1);
%% setup the model and the domain

% background wavespeed
c = ones(N,N);
 
% size of the model in interior domain
nxi  = size(c,2);
nyi  = size(c,1);
ni   = nxi*nyi;

xi = h*(0:nxi-1) - 0.5;
yi = h*(0:nyi-1) - 0.5;

[Xi,Yi] = meshgrid(xi,yi);

% size of the simulation domain
npml = 20;
nx = nxi + 2*npml;
ny = nyi + 2*npml;

x  = [xi(1)+(-npml:-1)*h xi xi(end)+(1:npml)*h];
y  = [yi(1)+(-npml:-1)*h yi yi(end)+(1:npml)*h];

[X,Y] = meshgrid(x,y);

% order of accuracy
order = 8;

% intensity of the pml absorbtion
sigmaMax = 80;
%%
pad = 1;
% we define the gaussian smoother
gaus = exp(-(Xi.^2 + Yi.^2)/0.000025);
% we smoothen the random field using a convolution
        
window = exp(-0.005./(0.2304-(Xi.^2+Yi.^2))).*(sqrt(Xi.^2+Yi.^2)<0.48);
window(isnan(window)) = 0;

%%
n = 5;
% the size of the data
Ntheta = 80;
% the number of angles of the sources and detectors (they are set to be the same).
dtheta = 2*pi/(Ntheta);

theta = linspace(0, 2*pi-dtheta, Ntheta);
d = [cos(theta).' sin(theta).'];

theta_r = linspace(0, 2*pi-dtheta, Ntheta);
r = [cos(theta_r).' sin(theta_r).'];

points_query = 0.5*r;
project_mat = zeros(Ntheta, nx, ny);

for ii = 1:nx
    for jj = 1:ny
        mat_dummy = zeros(nx,ny);
        mat_dummy(ii,jj) = 1;
        project_mat(:,ii,jj) = interp2(x,y,...
                                   reshape(mat_dummy, nx, ny),...
                                   points_query(:,1),...
                                   points_query(:,2));
    end
end

% properly reshaping and making it sparse
project_mat = sparse(reshape(project_mat, Ntheta, nx*ny));
 
eta80 = zeros(N^2, n);
scatter_real_freq_3 = zeros(Ntheta^2, n);
scatter_imag_freq_3 = zeros(Ntheta^2, n);
scatter_real_freq_5 = zeros(Ntheta^2, n);
scatter_imag_freq_5 = zeros(Ntheta^2, n);
scatter_real_freq_10 = zeros(Ntheta^2, n);
scatter_imag_freq_10 = zeros(Ntheta^2, n);
%%
omega3 = 2*2.5*pi;
omega5 = 2*5*pi;
omega10 = 2*10*pi;
U_in3 =  exp(1i*omega3*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
U_in5 =  exp(1i*omega5*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
U_in10 =  exp(1i*omega10*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
%%   
parfor i = 1:n
    i
    tic
    %generate a random Shepp-Logan phantom.
    eta = random_Shepp_Logan(nxi-2*pad,{'pad', pad; 'M', 1; 'phantom','msl'});
    eta = reshape(eta, nxi, nyi);
    smooth = conv2(gaus, eta, 'same'); 
    smooth = smooth/max(max(abs(smooth)));
    eta = smooth.*window;
    
    eta80(:,i) = reshape(eta, N^2, 1);

    m = 1 + eta;  
    eta_ext = ExtendModel(eta,nxi,nyi,npml);
    mext = ExtendModel(m,nxi,nyi,npml);
    
    H3 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega3,'compact_explicit');
    H5 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega5,'compact_explicit');
    H10 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega10,'compact_explicit');
    
    
    % building the right hand sides
    S3 = bsxfun(@times, -omega3^2*eta_ext, U_in3);
    S5 = bsxfun(@times, -omega5^2*eta_ext, U_in5);
    S10 = bsxfun(@times, -omega10^2*eta_ext, U_in10);
    
    % solving the equation
    U3 = H3\S3;
    U5 = H5\S5;
    U10 = H10\S10;
    
    % this is our "real data"
    scatter3 = project_mat*U3;
    scatter5 = project_mat*U5;
    scatter10 = project_mat*U10;
    
%     % adding noise
%     scatter = scatter.*(1 + randn(nxi,nyi));
    scatter_real_freq_3(:,i) = real(reshape(scatter3, Ntheta^2, 1));
    scatter_imag_freq_3(:,i) = imag(reshape(scatter3, Ntheta^2, 1));
    scatter_real_freq_5(:,i) = real(reshape(scatter5, Ntheta^2, 1));
    scatter_imag_freq_5(:,i) = imag(reshape(scatter5, Ntheta^2, 1));
    scatter_real_freq_10(:,i) = real(reshape(scatter10, Ntheta^2, 1));
    scatter_imag_freq_10(:,i) = imag(reshape(scatter10, Ntheta^2, 1));
    toc
end

%%
reshape_matrix = @(M, size, batch_size) permute(reshape(M, [size, size, batch_size]), [3, 1, 2]);

eta80 = reshape_matrix(eta80, N, n);
scatter_real_freq_2pt5 = reshape_matrix(scatter_real_freq_3, N, n);
scatter_real_freq_5 = reshape_matrix(scatter_real_freq_5, N, n);
scatter_real_freq_10 = reshape_matrix(scatter_real_freq_10, N, n);
scatter_imag_freq_2pt5 = reshape_matrix(scatter_imag_freq_3, N, n);
scatter_imag_freq_5 = reshape_matrix(scatter_imag_freq_5, N, n);
scatter_imag_freq_10 = reshape_matrix(scatter_imag_freq_10, N, n);

save(['inverse_scattering_shepplogan', num2str(n), '.mat'], ...
    'eta80', ...
    'scatter_real_freq_2pt5', 'scatter_real_freq_5', 'scatter_real_freq_10', ...
    'scatter_imag_freq_2pt5', 'scatter_imag_freq_5', 'scatter_imag_freq_10')

%%
% name = 'scatter80.h5';
% h5create(name, '/scatter_imag_freq_2.5',[Ntheta^2, n]);
% h5create(name,'/scatter_imag_freq_5',[Ntheta^2, n]);
% h5create(name,'/scatter_imag_freq_10',[Ntheta^2, n]);
% h5create(name,'/scatter_real_freq_2.5',[Ntheta^2, n]);
% h5create(name,'/scatter_real_freq_5',[Ntheta^2, n]);
% h5create(name,'/scatter_real_freq_10',[Ntheta^2, n]);
% %%
% h5write(name, '/scatter_real_freq_2.5', scatter_real_freq_3);
% h5write(name, '/scatter_imag_freq_2.5', scatter_imag_freq_3);
% h5write(name, '/scatter_real_freq_5', scatter_real_freq_5);
% h5write(name, '/scatter_imag_freq_5', scatter_imag_freq_5);
% h5write(name, '/scatter_real_freq_10', scatter_real_freq_10);
% h5write(name, '/scatter_imag_freq_10', scatter_imag_freq_10);
% 
% %%
% h5create('eta80.h5','/eta',[N^2, n]);
% h5write('eta80.h5','/eta',eta80);

%%
% check the data
figure(1);
imagesc(squeeze(eta80(1, :, :)));

figure(2); clf();
subplot(1,3,1);
imagesc(squeeze(eta80(1, :, :)));
title('eta');
subplot(1,3,2);
imagesc(squeeze(scatter_real_freq_2pt5(1, :, :)));
title('2.5 real');
subplot(1,3,3);
imagesc(squeeze(scatter_real_freq_5(1, :, :)));
title('5 imaginary');

figure(3); imagesc(squeeze(eta80(1, :, :)));