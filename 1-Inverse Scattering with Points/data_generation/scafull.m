% code to generate the scattering dataset, obtained from the authors of https://arxiv.org/abs/1810.09675
if(1)
    K=9; %number of waves
    NPW=9; % numer of grids per wave
    NPML=8; % number of PMLs at the boundary of the domain
    CPML=10; % PML parameter for boundary PMLs
    omega = 2*pi*K;
    
    %num of grid points
    NC = K*NPW; %number of cells
    h = 1/(NC+1);
    N=NC; N1=N;N2=N;
    Nsample = 5;
    
    %grid and window
    [x1,x2]=ndgrid(h*(1:N));    x1 = x1-1/2;    x2 = x2-1/2;
    %dis = min(1/2-abs(xx),1/2-abs(yy));    win = 1-exp(-(dis)^2);
    
    %PML
    gs=(1/NPML)*(0.5:0.5:NPML-0.5); eta=NPML*h;
    sigR=CPML/eta*gs.^2;
    sR=1./(1+1i*sigR/K);sL=sR(end:-1:1);
    s1=[sL,ones(1,2*(N1-2*(NPML-1))-1),sR];
    s2=[sL,ones(1,2*(N2-2*(NPML-1))-1),sR];    %nb = 2*NPML;
    
    %num directions
    Ndir = K*NPW;
    thetas = [0:Ndir-1]/Ndir * 2*pi;
    UI = zeros(N,N,Ndir);
    for b=1:Ndir
        tmp = thetas(b);
        t1 = cos(tmp);        t2 = sin(tmp);
        uinc = exp(i*omega*(x1*t1 + x2*t2));
        UI(:,:,b) = uinc;
    end
    UI = reshape(UI, N^2, Ndir);
    
    coes = zeros(N,N,Nsample);
    sols = zeros(Ndir,Ndir,Nsample);
    adjs = zeros(N,N,Nsample);
    
    gaussian = @(u1,u2,T,x1,x2) exp(-((x1-u1).^2+(x2-u2).^2) / (2*T));
    mag = 0.2;
    for idx=1:Nsample
        idx
        tic;
        %get media
        ng = 4;
        aux = zeros(size(x1));
        for gi=1:ng
            a=-0.3;            b=0.3;            u1 = a + rand(1)*(b-a);
            a=-0.3;            b=0.3;            u2 = a + rand(1)*(b-a);
            aux = aux + mag*gaussian(u1,u2,0.015^2,x1,x2);
        end
        m = aux;
        eta = -omega^2*m; %KEY: eta is the coefficient
        
        %get matrix
        c = sqrt(1./(1-m));
        ksq=(omega./c).^2;
        A=setupL2d(h,ksq,s1,s2);
        
        %solve
        DS = spdiags(eta(:), 0, N^2,N^2);
        U = A\(DS*UI);
        UT = U + UI; %total wave
        R = 1/(2*pi)*UI'*(DS*UT) * h^2;
        toc;
        
        %store data
        coes(:,:,idx) = eta;
        sols(:,:,idx) = R;
    end
    
    [pa,qa] = ndgrid(thetas);
    p1 = cos(pa);    p2 = sin(pa);
    q1 = cos(qa);    q2 = sin(qa);
    p1 = p1(:);    p2 = p2(:);
    q1 = q1(:);    q2 = q2(:);
    M = 1/(2*pi) * (exp(-i*omega*((p1-q1).*x1(:)'+(p2-q2)*x2(:)') ));
    
    adjs = (M'*(2*pi/Ndir)^2) * reshape(sols,[Ndir^2,Nsample]);
    adjs = reshape(adjs, [N,N,Nsample]);
    if 1
    for idx=1:Nsample
        subplot(1,3,1); imagesc(coes(:,:,idx)); colorbar;
        subplot(1,3,2); imagesc(real(sols(:,:,idx))); colorbar;
        subplot(1,3,3); imagesc(real(adjs(:,:,idx))); colorbar;
        pause(0.5);
    end
    end
end


if 1

coes = coes - mean(coes(:));
sols = sols - mean(sols(:));
adjs = adjs - mean(adjs(:));
 

coes = coes/(max(abs(coes(:))));
sols = sols/(max(abs(sols(:))));
adjs = adjs/(max(abs(adjs(:))));

end

if(1)
    filename  = 'scafull.mat';
    a = permute(real(coes), [3,1,2]);
    u = permute(real(sols), [3,1,2]);
    ai = permute(imag(coes), [3,1,2]);
    ui = permute(imag(sols), [3,1,2]);
    save(filename, 'a', 'u', 'ai', 'ui', '-v7.3');
end
