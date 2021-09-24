%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Author: Alberto Padoan                                                 %
%  Date: 24/09/2021                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Description %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script a illustrates a the model reduction framework presented in:
% 
% A. Padoan - "On model reduction by least squares moment matching"
%
% Example: Flexible Space Structure from https://tinyurl.com/vgpeaf7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc                                           
set(0, 'DefaultLineLineWidth', 1.5);          
% System parameters 
k = 30;                                       % Modes
m = 1;                                        % Actuators
q = 1;                                        % Sensors
% Model reduction parameters 
omega = [1e-2, 1e-1, 1e0, 1e1, ...            %
         5.5, 20, 16, 30, ...                 %
         50, 1e2, 1e3, 1e4];                  % Interpolation points
r =10;                                        % Order of least squares ROM
% Simulation parameters 
ns = 500;                                     % Samples
t_i = 0;                                      % Initial time
t_f = 10;                                     % Final time
tspan = linspace(t_i,t_f,ns);                 % Time span

%% Main

% Define FSS model
[A,B,C] = fss(k,m,q);                         % 
sys=ss(A,B,C,0);                              %
n = length(A);                                %

% Define signal generator
[S,w0,L] = sig_gen(omega);                    %
s_gen=ss(S,w0,L,0);                           % 
nu = length(S);                               %

% Solve the Sylvester equation A*Pi+BL = Pi*S
Pi = sylvester(A,-S,-B*L);                    % 

% Select Delta to assign the dominant eigenvalues 
e = esort(pole(sys));                         % Sort eigenvalues  
Delta = place(S',L',e(1:nu,:)).';             % Assign dominant eigenvalues 

% Select P to preserve eigenvalues and Q for least squares moment matching
[E,D,~] = eig((S-Delta*L)');                  % Compute eigenvalues
[~,idx] = esort(diag(D));                     % Sort eigenvalues
E= E(:,idx);                                  % Permute eigenvectors
P = [];                                       % Select P to preserve
for k=1:r/2                                   % dominant eigenvalues
   P = [P;  ...                               % 
             real(E(:,2*k-1)+E(:,2*k))'; ...  %
             imag(E(:,2*k-1)-E(:,2*k))'];     %
end                                           %
Q = pinv(P);                                  % Select Q to achieve LS MM
 
% Define least squares reduced order model
F = P*(S-Delta*L)*Q;                          % 
G = P*Delta;                                  % 
H = C*Pi*Q;                                   %  
rom = ss(F,G,H,0);                            % Least squares ROM

% % Alternative: compute F,G,H directly using convex optimization
% % Requires: cvx  
% cvx_begin
%     variable F(r,r)
%     variable G(r,1)
%     variable H(1,r)
%     minimize(norm(C*Pi-H*P,2))
%     subject to
%         F*P+G*L == P*S; 
% cvx_end
% rom = ss(F,G,H,0);                            % Least squares ROM

% Time simulations
[t,w] = ode45(@(t,w) S*w,tspan,w0);           % Simulate signal generator
t = t';  w = w';                              % Signal generator state
e_ss = (C*Pi-H*P)*w;                          % Steady-state error response  
rms_error = rms(e_ss)                         % Compute rms error bound
error_bound = norm(C*Pi-H*P)                  % Compute rms error bound

%% Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define and position figures
f1=figure(1); movegui(f1,'west');             % Fig1: System + ROM
f2=figure(2); movegui(f2,'center');           % Fig2: Error
f3=figure(3); movegui(f3,'east');             % Fig3: Time simulation

% Generate Bode diagrams
w = logspace(-2,4,2000);
[mag_sys,~,~] = bode(sys,w); 
[mag_rom,~,~] = bode(rom,w);
[mag_err_rom,~,~] = bode(minreal(sys-rom),w);

% Bode plots and time simulations
figure(1); loglog(w,squeeze(mag_sys),'k',w,squeeze(mag_rom),'b--');        grid on; hold on 
figure(2); loglog(w,squeeze(mag_err_rom)./squeeze(mag_sys),'b');           grid on; hold on  
figure(3); plot(t, e_ss, 'r'); xlabel('t'); ylabel('e_{ss}');              grid on; hold on

%% Save data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print in .csv file
csvwrite('FSS_sys.csv', [squeeze(w'), squeeze(mag_sys)]);
csvwrite('FSS_rom.csv', [squeeze(w'), squeeze(mag_rom)]);
csvwrite('FSS_err_rom.csv', [squeeze(w'), squeeze(mag_err_rom)./squeeze(mag_sys)]);

%% Functions

function [A,B,C] = fss(K,M,Q)
    rand('seed',1009);
    xi = rand(1,K)*0.001;       % Sample damping ratio
    omega = rand(1,K)*100.0;	% Sample natural frequencies
    A_k = cellfun(@(p) sparse([-2.0*p(1)*p(2),-p(2);p(2),0]), ...
                  num2cell([xi;omega],1),'UniformOutput',0);
    A = blkdiag(A_k{:}); 
    A = full(A);                                  
    B = kron(rand(K,M),[1;0]);
    C = 10.0*rand(Q,2*K);
end

function [S,w0,L] = sig_gen(w)
% SIG_GEN  Create a signal generator with eigenvalues at w(:).
%   [S,w0,L] = sig_gen(w) create a signal generator with given eigenvalues.
    if  any(w(:) == 0)
        w(w~=0);
    end
    S_k = cellfun(@(p) sparse(rjf(0,p(1))), ...
                  num2cell(w),'UniformOutput',0);
    S = blkdiag(S_k{:});
    S = full(S);
    L = [];
    if  any(w(:) == 0)
        S=blkdiag(0,S);
        L = [1];
    end
    nu = 2*length(S_k);
    %T = toeplitz(mod(1:nu,2));
    %L  = [L T(1,:)];
    L = ones(1,nu);
    L  = L/norm(L);
    w0  = flip(L)';
end

function M = rjf(sigma,omega) 
% RJF  Creates 2x2 matrix (in real Jordan canonical form) associated with
%      the eigenvalues sigma+i*omega and sigma-i*omega.
    M = [sigma, omega; -omega, sigma]; 
end













