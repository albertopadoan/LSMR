%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Author: Alberto Padoan                                                 %
%  Date: 06/10/2021                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Description %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script illustrates the nonlinear least squares model reduction 
% framework presented in:
% 
% A. Padoan - "Model reduction by least squares moment matching ... 
%              for  nonlinear systems"
%
% Example: Nonlinear inverter chain inspired by
%   C. Gu - "Model Order Reduction of Nonlinear Dynamical Systems"
% See also:
% - https://tinyurl.com/apnakyny
% - https://www2.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-217.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc        
set(0, 'DefaultLineLineWidth', 1.5);          
% System parameters 
global Vt tau Vdd alpha
n = 12;                                       % System order = # inverters
Vt = 1/5;                                     % Voltage threshold 
Vdd = 1./(4*(1:n));                           % Supply voltages 
tau = 4*(1:n);                                % Time constants
alpha = 1/4;                                  % Generator parameter
% Signal generator parameter
nu = 10;                                      % Signal generator order
Omega = 1;                                    % Angular frequency
% Reduced order model (ROM) parameters
r =4;                                         % LS ROM order (r < 2*nu)
% Simulation parameters 
ns = 500;                                     % Samples
t_i = 0;                                      % Initial time
t_f = 20;                                     % Final time
tspan = linspace(t_i,t_f,ns);                 % Time span
x0 = zeros(n,1);                              % Initial condition system
xi0 = zeros(r,1);                             % Initial condition ROM
w = logspace(-2,4,2000);                      % Bode plots frequencies

%% Main

% Define signal generator
[S,w0,L] = sig_gen(nu,Omega);                 % 
s_gen=ss(S,w0,L,0);                           % 

% Define nonlinear inverter chain 
[A,B,C,A3] = NL_inverter_chain(n,Vt);         % 
sys=ss(A,B,C,0);                              %

% Solve the Sylvester equation 
% (1)  A*Pi+BL = Pi*S
Pi = sylvester(A,-S,-B*L);                    % Solution of (1)

% Solve the Sylvester equation 
% (2)  A*Pi3+A3(Pi@Pi@Pi)N3 = Pi3*M3*(S@I^(2)+I@S@I+I^(2)@S)N3
% where A@B is the Kronecker product of A and B.
I=eye(nu); I2=kron(I,I); [M3, N3]= MN(nu,3);  % Auxiliary matrices ...
S3 = kron(S,I2)+kron(I,kron(S,I))+kron(I2,S); % ... to streamlinea notation
Pi3 = sylvester(A,-M3*S3*N3,...               % 
      -A3*(kron(Pi,kron(Pi,Pi)))*N3);         % Solution of (2)

% Select Delta to assign the dominant eigenvalues 
e = esort(pole(sys));                         % Sort eigenvalues  
Delta = place(S',L',e(1:nu,:)).';             % Assign dominant eigenvalues 

% Define linear approximation of surrogate reduced order model
F_bar = S-Delta*L;                            %
G_bar = Delta;                                %
H_bar = C*Pi;                                 %
rom_sur = ss(F_bar,G_bar,H_bar,0);            % Surrogate ROM

% Select P to preserve eigenvalues and Q for least squares moment matching
[E,DI,~] = eig((S-Delta*L)');                 % Compute eigenvalues
[~,idx] = esort(diag(DI));                    % Sort eigenvalues
E= E(:,idx);                                  % Permute eigenvectors
P = [];                                       % Select P to preserve ...
for k=1:r/2                                   % ... r dominant eigenvalues
   P = [P;  ...                               % 
             real(E(:,2*k-1)+E(:,2*k))'; ...  %
             imag(E(:,2*k-1)-E(:,2*k))'];     %
end                                           %
Q = pinv(P);                                  % Select Q to achieve LS MM
 
% Define approximate least squares reduced order model
F = P*F_bar*Q;                                % 
G = P*G_bar;                                  % 
H = H_bar*Q;                                  %
H3 = C*Pi3*M3;                                % 
rom = ss(F,G,H,0);                            % Linear least squares ROM

% Time simulations
[t,chi] = ode23(@(t,chi) ...                  %
            system(t,chi,sys,s_gen),...       % System + signal generator       
              tspan,[w0; x0]);                % 
t = t';  chi = chi';                          %
u = L*chi(1:nu,:);                            % Input
y_ss = C*chi(nu+1:end,:);                     % Steady-state output system
[~,zeta] = ode23(@(t,zeta) ...                %
            model(t,zeta,rom,s_gen),...       % ROM + signal generator       
              tspan,[w0; xi0]);               %
zeta = zeta';                                 %
y_ss1 = H*zeta(nu+1:end,:);                   % Steady-state output ROM1 
y_ss3 = H*zeta(nu+1:end,:)+...                % Steady-state output ROM3
        H3*kron_col(Q*zeta(nu+1:end,:),3);    % 
e_ss1 = y_ss - y_ss1;                         % Error ROM1
e_ss3 = y_ss - y_ss1;                         % Error ROM3

% Steady-state errors, rms errors, max rms errors
rms_u = rms(u);                               % Steady-state error ROM1 
rms_error1 = rms(e_ss1);                      % RMS error ROM1
gamma_rms1 = rms_error1/rms_u                 % Error bound ROM1
max_error1 = max(abs(e_ss1))                  % Max error ROM1 
e_ss3 = abs(y_ss - y_ss3);                    % Steady-state error ROM3
rms_error3 = rms(e_ss3);                      % RMS error ROM3
gamma_rms3 = rms_error3/rms_u                 % Error bound ROM3
max_error3 = max(abs(e_ss3))                  % Max error ROM3

%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define and position figures
f1=figure(1); movegui(f1,'northwest');        % Fig1: System+ROM1
f2=figure(2); movegui(f2,'southwest');        % Fig2: Error: Bode plot
f3=figure(3); movegui(f3,'northeast');        % Fig3: System+Surrogate ROM1
f4=figure(4); movegui(f4,'southeast');        % Fig4: Error: Bode plot
f5=figure(5); movegui(f5,'north');            % Fig5: Steady-state error 
f6=figure(6); movegui(f6,'south');            % Fig6: Steady-state output

% Generate Bode diagrams
[mag_sys,~,~] = bode(sys,w); 
[mag_rom,~,~] = bode(rom,w);
[mag_rom_sur,~,~] = bode(rom_sur,w);
[mag_err_rom,~,~] = bode(minreal(sys-rom),w);
[mag_err_rom_sur,~,~] =  bode(minreal(sys-rom_sur),w);

% Bode plots and time simulations
figure(1); loglog(w,squeeze(mag_sys),'k',w,squeeze(mag_rom),'b--');        
grid on; hold on; title('System vs ROM') 
figure(2); loglog(w,squeeze(mag_err_rom)./squeeze(mag_sys),'b');           
grid on; hold on; title('System vs ROM: error')  
figure(3); loglog(w,squeeze(mag_sys),'k',w,squeeze(mag_rom_sur),'r-.');    
grid on; hold on;  title('System vs Surrogate ROM') 
figure(4); loglog(w,squeeze(mag_err_rom_sur)./squeeze(mag_sys),'r');       
grid on; hold on;  title('System vs Surrogate ROM: error')   
figure(5); plot(t, e_ss1, 'r:'); xlabel('t'); ylabel('e_{ss}');              
grid on; hold on; 
figure(5); plot(t, e_ss3, 'b--'); xlabel('t'); ylabel('e_{ss}');              
grid on; hold on; title('Steady-state error: ROM1 and ROM3')
figure(6); plot(t, y_ss, 'k-'); xlabel('t'); ylabel('y_{ss}');              
grid on; hold on
figure(6); plot(t, y_ss1, 'r:'); xlabel('t'); ylabel('y_{ss}');         
grid on; hold on
figure(6); plot(t, y_ss3, 'b--'); xlabel('t'); ylabel('y_{ss}');         
grid on; hold on; title('Steady-state output: system, ROM1, ROM3')   

%% Save data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print in .csv file
csvwrite('nl_inverter_chain_sys.csv' , [t', y_ss']);
csvwrite('nl_inverter_chain_rom1.csv', [t', y_ss1']);
csvwrite('nl_inverter_chain_rom3.csv', [t', y_ss3']);
csvwrite('nl_inverter_chain_err1.csv', [t', e_ss1']);
csvwrite('nl_inverter_chain_err3.csv', [t', e_ss3']);

%% Functions

function dchidt = system(t,chi,sys,s_gen)
% SYSTEM Function handle which defines the interconnection of the nonlinear
    % inverter chain model with the signal generator.    
global Vt tau Vdd
n = length(sys.A); B =sys.B; 
nu = length(s_gen.A); S = s_gen.A; L = s_gen.C; 
dchidt = [S*chi(1:nu);
          -diag(tau)*diag(Vdd)*tanh((1/Vt)*[0; chi(nu+1:nu+n-1)])+...
          -diag(tau)*chi(nu+1:nu+n)+B*L*chi(1:nu)];
end

function dzetadt = model(t,zeta,rom,s_gen)
% MODEL Function handle which defines the interconnection of the nonlinear
    % inverter chain reduced order model with the signal generator.
r = length(rom.A);    F =rom.A;    G =rom.B; 
nu = length(s_gen.A); S = s_gen.A; L = s_gen.C; 
dzetadt = [S*zeta(1:nu);
          F*zeta(nu+1:nu+r)+G*L*zeta(1:nu)];
end

function [A,B,C,A3] = NL_inverter_chain(n,Vt)
% NL_INVERTER_CHAIN  Creates third order approximation of inverter chain 
%   [A,B,C,A3] = NL_inverter_chain(n,Vt) creates third order approximation 
    % of nonlinear inverter chain model, where the triple (A,B,C) defines
    % the linear approximation, while A3 defines the 3rd order term.
    global tau Vdd alpha
    A = -diag(tau) - diag(tau)*diag(Vdd)*diag((1/Vt)*ones(1,n-1),-1);
    B = tau(1)*alpha*eye(n,1);   % Used alpha*tau(1) = 1
    C = flip(eye(n,1))';
    A3 = zeros(n,n^3);
    for i=1:n-1
      A3(i+1,:) = (1/Vt)^3*double(1:n^3==(i-1)*(n^2+n+1)+1)/factorial(3);
    end
end

function [S,w0,L] = sig_gen(nu,Omega)
% SIG_GEN  Create a signal generator of order nu
%   [S,w0,L] = sig_gen(nu,Omega) creates a signal generator of order nu and
%   base frequency Omega.

    % Check that nu is integer and larger than 0
    nu =double(int32(nu));
    Omega =double(Omega);
    if (nu < 1)
        error('nu must be larger than 0')
    end
    % Check that Omega is positive
    if (Omega <=0 )
        error('Omega must be positive')
    end
    
    % Create a vector w of frequencies
    if rem(nu,2)==0
        % If nu even, create a (nu/2)-vector w of frequencies
        w =zeros(1,nu/2);
        for i =1:nu/2
           w(i) = i*Omega;
        end
    else
        % If nu odd, create a (ceil(nu/2))-vector w of frequencies
        if nu ==1
            w = 0;
        else
            w =zeros(1,ceil(nu/2));
            for i =2:ceil(nu/2)
               w(i) = (i-1)*Omega;
            end
        end
    end
    
    % Create signal generator matrices
    S = [];
    if  any(w(:) == 0)
        w = w(w~=0);
        S = 0;
    end
    S_k = cellfun(@(p) sparse(rjf(0,p(1))), ...
                  num2cell(w),'UniformOutput',0);
    S = blkdiag(S,full(blkdiag(S_k{:})));
    L = ones(1,length(S))/norm(ones(1,length(S)));
    w0  = flip(L)';
end

function M = rjf(sigma,omega) 
% RJF  Creates 2x2 matrix (in real Jordan canonical form) associated with
%      the eigenvalues sigma+i*omega and sigma-i*omega.
    M = [sigma, omega; -omega, sigma]; 
end

function [Mk, Nk] = MN(nu,k) 
% MN  Creates matrices Mk and Nk of order k as defined in Chapter 4 of
    %   J . Huang - "Nonlinear Output Regulation"
    alphabet = 'abcdefghijklmnopqrstuvwxyz'; 
    alphabet = alphabet(1:nu)';
    str_out = kron_str(alphabet,alphabet,k);
    str_out_red  = str_red(str_out);
    
    % Construct Nk
    Nk = zeros(nu^k,nchoosek(nu+k-1,k));
    for i=1:nu^k
        for j=1:nchoosek(nu+k-1,k)
            if (sort(str_out(i,:)) == str_out_red(j,:))
                Nk(i,j) = 1;
            end
        end
    end
    % Normalize each column
    for j=1:nchoosek(nu+k-1,k)
                Nk(:,j) = Nk(:,j)/norm(Nk(:,j));
    end
    Mk = Nk';
end

function str_out  = kron_str(alphabet,str_in,k) 
% KRON_STR  Auxiliary function for the function MN
    % Base case
    if k<=1
        str_out = str_in;
        return 
    end
    % Recursion
    prev = kron_str(alphabet,str_in,k-1);
    % Compute full
    str_out = [];
    for i=1:length(alphabet)
        str_out = [str_out; strcat(alphabet(i),prev)];
    end
end

function str_out_red  = str_red(str_out) 
% STR_RED Auxiliary function for the function MN
    str_out_red = str_out;
    for i=1:length(str_out_red)
        str_out_red(i,:) = sort(str_out_red(i,:));
    end
    str_out_red = unique(str_out_red,'rows');
end

function vK = kron_col(v,K)
% KRON_COL Auxiliary function for the function MN
    [I,J] = size(v);
    vK = zeros(I^3,J);
    temp = [];
    for j=1:J
        temp = v(:,j);
        for k=1:K-1 
            temp = kron(v(:,j),temp); 
        end
        vK(:,j)= temp;
        temp = [];
    end
end



