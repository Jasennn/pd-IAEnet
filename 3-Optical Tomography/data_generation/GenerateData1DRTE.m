function GenerateData1DRTE
% this code solves RTE on a slab geometry
% the computational domain is [0,1]x[-1,1]


%% initialization
% global Nr Nv dr v0 sigma_x
Nr = 49;    % discretization on space domain
Nv = 50;    % discretization on velocity domain
dr = 1/Nr;
dv = 2/Nv;
v0 = -1+dv/2:dv:1-dv/2;
r0 = 0:dr:1;

N  = 2;    % number of training data pairs
% Data = cell(N,2); 
Data_sigma = zeros(N, 1, Nr+1);
Data_meas = zeros(N, Nv, Nv);

%% data generation

parfor n = 1:N % for parallelization
%for n = 1:N % for debugging
    tic 
    % generate input
    a = 1/2*rand;
    b = 4*rand + 1;
    sigma = @(x)MediaFun(x,a,b);
    sigma_x = sigma(r0);
    % Data{n,1} = sigma_x;
    Data_sigma(n, :, :) = sigma_x;

    % generate output
    meas = zeros(Nv,Nv);
    for i = 1:Nv
        inflow = zeros(Nv,1);
        inflow(i,1) = 1;
        outflow = LTE_1D_ep(inflow, Nr, Nv, dr, v0, sigma_x);
        meas(:,i) = outflow;
    end
    % Data{n,2} = meas;
    Data_meas(n, :, :) = meas;
    n
    toc
end

%% diagnostics
% sigma1 = squeeze(Data_sigma(1, :, :));
% sigma2 = squeeze(Data_sigma(2, :, :));
meas1 = squeeze(Data_meas(1, :, :));
meas2 = squeeze(Data_meas(2, :, :));
for k=1:2
    figure(2*k - 1); plot(squeeze(Data_sigma(k, :, :)));
    figure(2*k); imagesc(squeeze(Data_meas(k, :, :)));
end

rel_err = norm(meas2 - meas1) / norm(meas1) 

%% saving
save('RTE1D.mat', 'Data_sigma', 'Data_meas');

end

function sigma = MediaFun(x,a,b) 
% Media function A(x) = 0.1 + b*sin(2*pi(x-a))*chi(x,[a,a+1/2])
% where a~Uniform(0,1/2), b~Uniform(1,5)
% chi(x,[a,a+1/2]) is the indicator function on [a,a+1/2]
temp = b*sin(2*pi*(x-a));
temp = temp.*(temp>0);
sigma = 0.1 + temp;
end

function outflow = LTE_1D_ep(inflow, Nr, Nv, dr, v0, sigma_x)
% global Nr Nv

%% initial 
f = ones(Nv,Nr+1);
data_pre = zeros(size(f));
data_pre((Nv/2+1):end,1) = inflow((Nv/2+1):end, 1); % left-inflow
data_pre(1:Nv/2 , end) = inflow(1:Nv/2 , 1); % right-inflow
data_pre = data_pre(:);

func = @(p) MAB_multi(p, Nr, Nv, dr, v0, sigma_x);
[f, flag] = gmres(func,data_pre,250,1e-10);
f = reshape(f,Nv,Nr+1);

outflow = zeros(Nv,1);
outflow(1:Nv/2,1) = f(1:Nv/2,1); % left-ouflow
outflow((Nv/2+1):end,1) = f((Nv/2+1):end,end); % right-outflow

end

function Mb = MAB_multi(p, Nr, Nv, dr, v0, sigma_x)
% global Nr Nv dr v0 sigma_x
e = ones(Nv,1);
v = v0(:);

p = reshape(p,Nv,Nr+1);
Rp = zeros(size(p));
BCp = zeros(size(p));
boundary_D = zeros(size(p));

for kv = 1:Nv
    p_temp = p(kv,:);
    
    Rp_pre = (p_temp(2:end) - p_temp(1:end-1))/dr;
    
    if v(kv)>=0
        Rp(kv,2:end) = v(kv)*Rp_pre;
        boundary_D(kv,1) = p(kv,1);
    else
        Rp(kv,1:end-1) = v(kv)*Rp_pre;
        boundary_D(kv,end) = p(kv,end);
    end
end

BCp(:,2:Nr) = p(:,2:Nr) - e*e'*p(:,2:Nr)/Nv;
BCp = (ones(Nv,1)*sigma_x).*BCp;

Mb = Rp + BCp + boundary_D;

Mb = Mb(:);

end