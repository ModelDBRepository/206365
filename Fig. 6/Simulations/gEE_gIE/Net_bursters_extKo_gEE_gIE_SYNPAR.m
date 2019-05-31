% Created by Anatoly Yu. Buchin, August 2014 - April 2015
% Excitatory Bazhenov neuron
% Inhibitory Bazh neurons

%{
%% NETWORK ELEMETS
  %   Ne=100;   Ni=20;  % very small Network
      Ne=841; Ni=225;    % middle Network I
  %   Ne=2500; Ni=625;   % middle Network II
  %   Ne=8100; Ni=1936;   % large Network
%%  

%% INTEGRATION PARAMETERs
T=10;        % total time, mS
dt=0.05;       % time step, ms

Tframe=0;      % first frame for the movie
dTframe=10;
frame=1;

% constants
F=96489;        % coul/M
%%

%% CONNECTIVITY MATRICIES

% network one parameters
gEE_mean_NMDA=0.00002;                % mS/cm^2  0.00005

gEE_mean_AMPA=0.0015;                 % mS/cm^2   0.0018
gII_mean_GABA=0.0005;                 % mS/cm^2   0.0005
gIE_mean_GABA=0.0007;                 % mS/cm^2   0.001
gEI_mean_AMPA=0.001;                  % mS/cm^2   0.0015
% connection probabilities,          from Yangfan Peng poster FENS 2014
pEE=0.05;
pEI=0.3;
pIE=0.65;
pII=0.4;
%  generate connectivity matrix, with heterogenity
S_EE_AMPA=random('normal',gEE_mean_AMPA,0.1*gEE_mean_AMPA,Ne,Ne);     % EE connection AMPA
S_EE_NMDA=random('normal',gEE_mean_NMDA,0.1*gEE_mean_NMDA,Ne,Ne);     % EE connection NMDA
S_II_GABA=random('normal',gII_mean_GABA,0.1*gII_mean_GABA,Ni,Ni);     % II connection GABA
S_IE_GABA=random('normal',gIE_mean_GABA,0.1*gIE_mean_GABA,Ne,Ni);     % IE connection GABA
S_EI_AMPA=random('normal',gEI_mean_AMPA,0.1*gEI_mean_AMPA,Ni,Ne);     % EI connection AMPA

% FINAL CONNECTIVITY MATRIX, randomly set to zero non-connected elements 
if isempty(find(S_EE_AMPA)) == 0
S_EE_AMPA(randsample(find(S_EE_AMPA),round(Ne^2*(1-pEE))))=0;
end
if isempty(find(S_II_GABA)) == 0
S_II_GABA(randsample(find(S_II_GABA),round(Ni^2*(1-pII))))=0;
end
if isempty(find(S_IE_GABA)) == 0
S_IE_GABA(randsample(find(S_IE_GABA),round(Ni*Ne*(1-pIE))))=0;
end
if isempty(find(S_EI_AMPA)) == 0
S_EI_AMPA(randsample(find(S_EI_AMPA),round(Ne*Ni*(1-pEI))))=0;
end
if isempty(find(S_EE_NMDA)) == 0
S_EE_NMDA(randsample(find(S_EE_NMDA),round(Ne^2*(1-pEE))))=0;
end

% NMDA are in the same place where AMPA
% S_EE_NMDA=heaviside(S_EE_AMPA-0.0001).*S_EE_NMDA;           

% LOCATION OF IN in PY network
ind_Ko=randsample(1:1:Ne,Ni);               % random 
% [gmax_IE,ind_Ko]=max(S_IE_GABA);          % maximal conductance
%%
  
%
%% LFP MODEl
k=0.02;         % prop. coefficient, 2*pi*a*p*R1^2/sigma/R/ri
                % a - dendrite diameter
                % p - density of charges (pyramidal cells)
                % R1 - radius of a spheric layer
                % sigma - average conductivity of an extracellular matrix
                % R - distance of the electrode from the spherical layer                
%%
   
%% E POPULATION, PY BAZHENOV MODEL
% Mg
Mg=0.25;          % mM, Mg concentration for NMDA synapses
% CL
Clo_E=130;        % mM                                
Vhalf_E=40;                % KCC2 1/2
HCO3o_E=26;       % mM
HCO3i_E=16;       % mM
kCL_E=100;        % CL conversion factor, 1000 cm^3
% K
Ki_E=150;            % intra K, mM
kK_E=10;             % K conversion factor, 1000 cm^3
d_E=0.15;            % ratio Volume/surface, m

D_E(1:Ne,1)=4e-6;    % diffusion coefficient, cm^2/s  4e-6, mum^2/ms = cm^2/s*10-8    [Bazhenov 2004], [Fisher 1976] Ko the cat neocortex
D_E_slice(1:Ne,1)=4e-6/10;   % diffusion coefficient, cm^2/s  4e-6, mum^2/ms = cm^2/s*10-8 /10, x20 slower 1/tau = 1/2000ms
dx_E=50*1e-4;        % distance between volume centers, cm Huberfeld et al 2007 estimation
dx_ext_E=200*1e-4;   % distance between the cells and external volume
% NA
Nao_E=130;        % extra Na, mM
Nai_E=20;         % intra Na, mM
kNa_E=10;         % NA conversion, 1000 cm^3
% single cell and membrane parameters
Cm_E=0.75;         % mu F/cm^2
e0_E=26.6393;      % kT/F, in Nernst equation
kappa_E=10000;     % conductance between compartments, Ohm
S_Soma_E=0.000001; % cm^2
S_Dend_E=0.000165; % cm^2
% somatic conductances
G_Na_E=3450.0;     % Na, mS/cm^2
G_Kv_E=200.0;      % Kv-channel mS/cm^2
Vbolz_E=22;        % mV, activation constant for K-current
gg_kl_E=0.042;     % K leak, mS/cm^2
gg_Nal_E=0.0198;   % Na leak, mS/cm^2
% dendritic conductances
G_NaD_E=1.1;       % mS/cm^2
G_NapD_E=3.5;      % mS/cm^2
G_HVA_E=0.0195;    % mS/cm^2
G_kl_E=0.044;      % K leak, mS/cm^2
G_lD_E=0.01;     % Clleak, mS/cm^2
G_Nal_E=0.02;      % Na leak, mS/cm^2
E_Ca_E=140;        % mV
TauCa_E=1000;      % ms
DCa_E=0.85;        % mV ?
G_KCa_E=2.5;       % mS/cm^2
G_Km_E=0.01;       % mS/cm^2
% Pump parametes and glial buffer
Koalpha_E=3.5;    % mM
Naialpha_E=20;    % mM
Imaxsoma_E=25;    % muA/cm^2
Imaxdend_E=25;    % muA/cm^2
Kothsoma_E=15;    % mM
koff_E=0.0008;    % 1/ mM / ms  0.0008 in Bazh model            
K1n_E=1.0;        % 1/ mM
Bmax_E=500;       % mM
% Ko DIFFUSION  
 %Diffusion matrix
 A_diff=M_diff(sqrt(Ne)); 
 % bath Ko concentration
 Ko_bath=zeros(Ne,1);
 Ko_bath(1:end)=8; 
 % INDEXES OF ION COUPLED ELEMENTS
    l=zeros(Ne,4);
for i=1:1:Ne 
   l(i,1:length(find(A_diff(i,1:end)>0)))=find(A_diff(i,1:end)>0);
end
% INDEXES OF BORDER ELEMENTS
A_border=M_diff_border(sqrt(Ne));
q=zeros(Ne,4);
for i=1:1:Ne % find indexes of border elements
   q(i,1:length(find(A_border(i,1:end)>0)))=find(A_border(i,1:end)>0);
end    
[row,col,border_elements] = find(q);
border_elements=unique(border_elements);   % elements on the border, UNIQUE indexes

% NO PERIODIC conditions for diffusion
% D_E(border_elements)=0;                    
% EXTERNAL DIFFUSION from the border
%  D_E_slice(1:end,1)=0;
%  D_E_slice(border_elements,1)=4e-6/10;

%%

%% I POPULATION, IN BAZHENOV MODEL
% CL
Clo_I=130;        % mM
Cli_I=3.70;       % mM
Vhalf_I=40;       % KCC2 1/2
% K
Ki_I=150;         % intra K, mM
kK_I=10;          % K conversion factor, 1000 cm^3
Vbolz_I=22;        % constant for K delayed-rect current
d_I=0.15;         % ratio Volume/surface, mu m
% NA
Nao_I=130;        % extra Na, mM
Nai_I=20;         % intra Na, mM
% single cell and membrane parameters
Cm_I=0.75;         % mu F/cm^2
e0_I=26.6393;      % kT/F, in Nernst equation
kappa_I=10000;     % conductance between compartments, Ohm
S_Soma_I=0.000001; % cm^2
S_Dend_I=0.000050; % cm^2
% Conductances active
G_Na_I=3450.0;     % Na, mS/cm^2
G_Kv_I=200.0;      % Kv-channel mS/cm^2
% dendrite leaks
G_kl_I=0.035;      % mS/cm^2
G_lD_I=0.01;       % Cl, mS/cm^2
G_Nal_I=0.02;      % mS/cm^2
% somatic leaks
gg_kl_I=0.042;     % mS/cm^2
gg_Nal_I=0.0198;   % mS/cm^2
% Pump parametes and glial buffer
Koalpha_I=3.5;    % mM
Naialpha_I=20;    % mM
Imaxsoma_I=25;    % muA/cm^2
%%

%% SYNAPTIC PARAMETERS
% AMPA
alpha2_AMPA=0.185;
tau_AMPA=1/alpha2_AMPA;               % 5.4 ms
V_AMPA=0;                   % mV
A_E=1/dt;                    % delta-funciton appr, step increase for g_AMPA

% NMDA
tau_NMDA_rise=2;            % ms
tau_NMDA_decay=100;         % ms
alpha_NMDA=0.5;             % 1/ms
V_NMDA=0;                   % mV

% GABA-A
alpha2_GABA=0.120;
tau_GABA=1/alpha2_GABA;              % ms 8.33
V_GABA=-75;                 % mV
A_I=1/dt;                    % delta-funciton appr, step increase for g_GABA

% refractory period, same for E and I pop
t_ref=2;

%%
%}

% load the network parameters
load('Net_transition_1_30.mat');

% variation steps in %
dq=10;
dp=10;

% maximal amount of variation in %
qmax=150;
pmax=150;

% maximal number of steps
q_step=round(qmax/dq)+1;
p_step=round(pmax/dp)+1;

% axis for variation
Q_axis=(0:1:(q_step-1))*dq;
P_axis=(0:1:(p_step-1))*dp;

% matrix of maximal oscillaiton frequencies
maxFR=zeros(length(1:1:(q_step)),length(1:1:(p_step)));
ampFR=zeros(length(1:1:(q_step)),length(1:1:(p_step)));
simTime=zeros(length(1:1:(q_step)),length(1:1:(p_step)));

% create cell array for all the LFPs
aLFP=cell(length(0:dq:q_step),length(0:dp:p_step));

% store variation parameters in a temporary variable
S_EE_AMPA_fix=S_EE_AMPA;
S_IE_GABA_fix=S_IE_GABA;   


for q=1:1:q_step         % from 0 to qmax, so +1
    
parfor p=1:1:p_step
       
tic               % start the timer for one cell

Q=(q-1)*dq/100;
P=(p-1)*dp/100; % value for variation

%% INTEGRATION PARAMETERS
T=60000;       % total time in one cell, mS % should be 90000
dt=0.05;    % time step, ms

sw=5000;      % sweep size for LFP, ms
dsw=5000;     % sweep step for LFP, ms
Tr=35;      % threshould for amplitude peak =35 for seizure detection

%% SYNAPTIC VARIATION
% reduce the whole matrix (distribution of the synaptic parameters)
S_EE_AMPA=Q*S_EE_AMPA_fix;     % EE connection AMPA
S_IE_GABA=P*S_IE_GABA_fix;     % IE connection GABA

%S_EE_NMDA;     % EE connection NMDA
%S_II_GABA;     % II connection GABA
%S_EI_AMPA;     % EI connection AMPA

       
%% ICS NOISE
% synaptic noise to E population
CE=100;         % number of synapses per P neuron            100
SE_AMPA=0.5;     % average value of synaptic gating variable
tauE_AMPA=1/alpha2_AMPA; %
gE_AMPA=0.0005;  % conductance per external input uS/cm^2
VE_av=-60;      % numerical estimate of mean(VEnorm)
sigmaE=abs(gE_AMPA*sqrt(CE*SE_AMPA*tauE_AMPA)*(VE_av-V_AMPA));

% synaptic noise to I population
CI=150;          % number of synapses per I neuron            150
SI_AMPA=0.5;     % average value of synaptic gating variable
tauI_AMPA=1/alpha2_AMPA; %
gI_AMPA=0.0005;   % conductance per external input uS/cm^2
VI_av=-60;       % numerical estimate of mean(VEnorm)
sigmaI=abs(gI_AMPA*sqrt(CI*SI_AMPA*tauI_AMPA)*(VI_av-V_AMPA));
%%

%% ICs E POPULATION
 het_E=0.05;                % heterogenity of IC in E population

 % random ICs (KCC2 norm)
 Bs_E=random('normal',498.83,0,Ne,1);                         % mM buffer sould not be random
 Cli_E=subplus(random('normal',8,het_E*6.37,Ne,1));              % mM 5.13, Ikcc2=0.5
 Ko_E=subplus(random('normal',8,0,Ne,1));                     % mM
 cai_E=subplus(random('normal',0,0,Ne,1));                    % mM 
 VD_E=random('normal',-67.42,67.42*het_E,Ne,1);               % mV 
 VSOMA_E(Npath)=random('normal',-67.48,het_E*67.48,length(Npath),1);               % mV 
 m_iKv_E=subplus(random('normal',0.00,het_E*0.00,Ne,1));        % 1
 m_iNa_E=subplus(random('normal',0.01,het_E*0.01,Ne,1));        % 1
 h_iNa_E=subplus(random('normal',0.82,het_E*0.88,Ne,1));        % 1 
 m_iNaD_E=subplus(random('normal',0.01,het_E*0.01,Ne,1));       % 1
 h_iNaD_E=subplus(random('normal',0.88,het_E*0.88,Ne,1));       % 1
 m_iNapD_E=subplus(random('normal',0.00,het_E*0.00,Ne,1));      % 1 
 m_iKCa_E=subplus(random('normal',0.00,het_E*0.00,Ne,1));       % 1  
 m_iHVA_E=subplus(random('normal',0.00,het_E*0.00,Ne,1));       % 1
 h_iHVA_E=subplus(random('normal',0.61,het_E*0.61,Ne,1));       % 1
 m_iKm_E=subplus(random('normal',0.01,het_E*0.01,Ne,1));        % 1
 
  % random ICs (KCC2 path, population)
 Bs_E(Npath)=random('normal',499.92,0,length(Npath),1);                          % mM buffer sould not be random
 Cli_E(Npath)=subplus(random('normal',12,het_E*12.13,length(Npath),1));         % mM
 Ko_E(Npath)=subplus(random('normal',8,het_E*3.46,length(Npath),1));            % mM
 cai_E(Npath)=subplus(random('normal',0,het_E*0,length(Npath),1));                 % mM 
 VD_E(Npath)=random('normal',-63.18,63.18*het_E,length(Npath),1);                  % mV 
 VSOMA_E(Npath)=random('normal',-62.00,het_E*62.00,length(Npath),1);               % mV 
 m_iKv_E(Npath)=random('normal',0.00,het_E*0.00,length(Npath),1);                  % 1 
 m_iNa_E(Npath)=random('normal',0.02,het_E*0.00,length(Npath),1);                  % 1
 h_iNa_E(Npath)=random('normal',0.75,het_E*0.75,length(Npath),1);                  % 1 
 m_iNaD_E(Npath)=subplus(random('normal',0.02,het_E*0.02,length(Npath),1));        % 1
 h_iNaD_E(Npath)=subplus(random('normal',0.75,het_E*0.75,length(Npath),1));        % 1 
 m_iNapD_E(Npath)=subplus(random('normal',0.00,het_E*00,length(Npath),1));         % 1 
 m_iKCa_E(Npath)=subplus(random('normal',0.00,het_E*00,length(Npath),1));          % 1 
 m_iHVA_E(Npath)=subplus(random('normal',0.00,het_E*00,length(Npath),1));          % 1
 h_iHVA_E(Npath)=subplus(random('normal',0.54,het_E*0.54,length(Npath),1));        % 1
 m_iKm_E(Npath)=subplus(random('normal',0.02,het_E*0.02,length(Npath),1));         % 1
 % vector of refractory times, >t_ref
 VSOMA_E_sp=10*ones(Ne,1);
 
 % REPRESENTATIVE CELLS
 % PY, KCC2 norm
 VEnorm=zeros(1);
 VEnorm_IE=zeros(1);
 VEnorm_II=zeros(1);
 
 VEnorm_AMPA=zeros(1);
 VEnorm_GABA=zeros(1);
 
 Konorm=zeros(1);
 Clinorm=zeros(1);
 cai_VEnorm=zeros(1);
 % PY, KCC2 path
 VEpath=zeros(1);
 VEpath_IE=zeros(1);
 VEpath_II=zeros(1);
 
 VEpath_AMPA=zeros(1);
 VEpath_GABA=zeros(1);
 
 Kopath=zeros(1);
 Clipath=zeros(1);
 cai_VEpath=zeros(1);
 
 %% LFP variable %%%
 LFP=0;
 
 %% ICs I POPULATION
het_I=0.05;
VD_I=random('normal',-65.65,het_I*65.65,Ni,1);                   % mV
VSOMA_I=-65.57;    % mV
m_iKv_I=subplus(random('normal',0.00,het_I*0.00,Ni,1));    % 1;
m_iNa_I=subplus(random('normal',0.01,het_I*0.01,Ni,1));    % 1
h_iNa_I=subplus(random('normal',0.84,het_I*0.84,Ni,1));    % 1
% vector of refractory times, >t_ref
VSOMA_I_sp=10*ones(Ne,1);
% Representative cell
% IN
 VI1=zeros(1);
 VI1_IE=zeros(1);
 VI1_II=zeros(1);
 % clamp response
 VI_AMPA=zeros(1);
 VI_GABA=zeros(1);
 
%% Synaptic ICs
IE=0;
II=0; 
% EE, AMPA
IE_AMPA=zeros(Ne,1);
gEE_AMPA=zeros(Ne,1);
% EE, NMDA
gEE_NMDA=zeros(Ne,1);
x_NMDA=zeros(Ne,1);
% EI, AMPA
II_AMPA=zeros(Ni,1);
gEI_AMPA=zeros(Ni,1);
% II, GABA
gII_GABA=zeros(Ni,1);
% IE, GABA
gIE_GABA=zeros(Ne,1);
 % EXC Firings
firings_E=[];                                 % spike timings
fired_E=[];                                   % indexes of spikes fired at t
 % INH Firings
firings_I=[];                                 % spike timings
fired_I=[];                                   % indexes of spikes fired at t
%%

%% TIME INTEGRATION LOOP
for t=1:1:round(T/dt)                        
    
%% E POPULATION INTEGRATION
 % Na-P-pump
 Ap_E=(1/((1+(Naialpha_E/Nai_E))^3))*(1./((1+(Koalpha_E./Ko_E)).^2));
 Ikpump_E=-2*Imaxsoma_E*Ap_E;
 INapump_E=3*Imaxsoma_E*Ap_E; 
 % reversal potentials on soma and dendrite
 % K
 VK_E=e0_E*log(Ko_E/Ki_E);
 % NA
 VNA_E=e0_E*log(Nao_E/Nai_E);
 % CL
 VCL_E=e0_E*log(Cli_E/Clo_E);
 VGABA_E=e0_E*log((4*Cli_E+HCO3i_E)./(4*Clo_E+HCO3o_E));
 
 % dendrite current
 iDendrite_E= -G_lD_E*(VD_E-VCL_E) -G_kl_E*(VD_E-VK_E) -G_Nal_E*(VD_E-VNA_E) -2.9529*G_NaD_E*m_iNaD_E.^3.*h_iNaD_E.*(VD_E-VNA_E) -G_NapD_E*m_iNapD_E.*(VD_E-VNA_E) -G_KCa_E*m_iKCa_E.^2.*(VD_E-VK_E) -2.9529*G_HVA_E*m_iHVA_E.^2.*h_iHVA_E.*(VD_E-E_Ca_E) -2.9529*G_Km_E*m_iKm_E.*(VD_E-VK_E) -INapump_E -Ikpump_E +IE;

 % somatic voltage
 g1_SOMA_E=gg_kl_E +gg_Nal_E +(2.9529*G_Na_E*m_iNa_E.^3.*h_iNa_E) +(2.9529*G_Kv_E.*m_iKv_E);
 g2_SOMA_E=gg_kl_E*VK_E +gg_Nal_E*VNA_E +(2.9529*G_Na_E*m_iNa_E.^3.*h_iNa_E*VNA_E) +(2.9529*G_Kv_E*m_iKv_E.*VK_E) -INapump_E -Ikpump_E;
 VSOMA_E=(VD_E + (kappa_E*S_Soma_E *g2_SOMA_E)) ./ (1+kappa_E*S_Soma_E*g1_SOMA_E);
 
 % IKv Soma
 a_iKv_E=0.02*(VSOMA_E-Vbolz_E)./(1-exp(-(VSOMA_E-Vbolz_E)/9));
 b_iKv_E=-0.002*(VSOMA_E-Vbolz_E)./(1-exp((VSOMA_E-Vbolz_E)/9));
 tauKvm_E=1./((a_iKv_E+b_iKv_E)*2.9529);
 infKvm_E=a_iKv_E./(a_iKv_E+b_iKv_E);

 % INa Soma
 am_iNa_E=0.182*(VSOMA_E-10+35)./(1-exp(-(VSOMA_E-10+35)/9));
 bm_iNa_E=0.124*(-VSOMA_E+10-35)./(1-exp(-(-VSOMA_E+10-35)/9));
 ah_iNa_E=0.024*(VSOMA_E-10+50)./(1-exp(-(VSOMA_E-10+50)/5));
 bh_iNa_E=0.0091*(-VSOMA_E+10-75)./(1-exp(-(-VSOMA_E+10-75)/5));
 tau_m_E=(1./(am_iNa_E+bm_iNa_E))/2.9529;
 tau_h_E=(1./(ah_iNa_E+bh_iNa_E))/2.9529;
 m_inf_new_E=am_iNa_E./(am_iNa_E+bm_iNa_E);
 h_inf_new_E=1./(1+exp((VSOMA_E-10+65)/6.2));

 % NaP, D current
 minfiNapD_E = 0.02./(1 + exp(-(VD_E+42)/5));
 
 % INa D, sodium channel
 am_iNaD_E=0.182*(VD_E-10+35)./(1-exp(-(VD_E-10+35)/9));
 bm_iNaD_E=0.124*(-VD_E+10-35)./(1-exp(-(-VD_E+10-35)/9));
 ah_iNaD_E=0.024*(VD_E-10+50)./(1-exp(-(VD_E-10+50)/5));
 bh_iNaD_E=0.0091*(-VD_E+10-75)./(1-exp(-(-VD_E+10-75)/5));
 minf_newD_E = am_iNaD_E./(am_iNaD_E+bm_iNaD_E);
 hinf_newD_E = 1./(1+exp((VD_E-10+65)/6.2));
 tau_mD_E = (1./(am_iNaD_E+bm_iNaD_E))/2.9529;
 tau_hD_E = (1./(ah_iNaD_E+bh_iNaD_E))/2.9529;

%%%% iKCa %%%%
 minf_iKCa_E = (48*cai_E.^2/0.03)./(48*cai_E.^2/0.03 + 1);
 taum_iKCa_E = (1./(0.03*(48*cai_E.^2/0.03 + 1)))/4.6555;

%%%% IHVA %%%%
 am_iHVA_E = 0.055*(-27 - VD_E)./(exp((-27-VD_E)/3.8) - 1);
 bm_iHVA_E = 0.94*exp((-75-VD_E)/17);
 ah_iHVA_E = 0.000457*exp((-13-VD_E)/50);
 bh_iHVA_E = 0.0065./(exp((-VD_E-15)/28) + 1);
 tauHVAh_E = 1./((ah_iHVA_E+bh_iHVA_E)*2.9529);
 infHVAh_E = ah_iHVA_E./(ah_iHVA_E+bh_iHVA_E);
 tauHVAm_E = 1./((am_iHVA_E+bm_iHVA_E)*2.9529);
 infHVAm_E = am_iHVA_E./(am_iHVA_E+bm_iHVA_E);
 
 %%% IKM %%%%
 am_iKm_E = 0.001*(VD_E + 30)./(1 - exp(-(VD_E + 30)/9));
 bm_iKm_E = -0.001*(VD_E + 30)./(1 - exp((VD_E + 30)/9));
 tauKmm_E = 1./((am_iKm_E+bm_iKm_E)*2.9529);
 infKmm_E = am_iKm_E./(am_iKm_E+bm_iKm_E);

% GLIA
 kon_E=koff_E./(1+exp((Ko_E-Kothsoma_E)/(-1.15)));
 Glia_E=koff_E*(Bmax_E-Bs_E)/K1n_E -kon_E/K1n_E.*Bs_E.*Ko_E;
 
% PY population integration
 VD_E=((1/Cm_E).*(iDendrite_E +(VSOMA_E-VD_E) / (kappa_E*S_Dend_E)))*dt + VD_E;
 m_iNa_E=(-(m_iNa_E-m_inf_new_E)./tau_m_E)*dt + m_iNa_E;
 h_iNa_E=(-(h_iNa_E-h_inf_new_E)./tau_h_E)*dt + h_iNa_E;
 m_iKv_E=(-(m_iKv_E-infKvm_E)./tauKvm_E)*dt + m_iKv_E; 
 m_iNaD_E =(-(m_iNaD_E - minf_newD_E)./tau_mD_E)*dt +m_iNaD_E;
 h_iNaD_E =(-(h_iNaD_E - hinf_newD_E)./tau_hD_E)*dt +h_iNaD_E;
 m_iNapD_E=(-(m_iNapD_E - minfiNapD_E)./0.1992)*dt +m_iNapD_E; 
 m_iKCa_E =(-(1./taum_iKCa_E).*(m_iKCa_E - minf_iKCa_E))*dt + m_iKCa_E;
 m_iHVA_E = (-(m_iHVA_E-infHVAm_E)./tauHVAm_E)*dt + m_iHVA_E;
 h_iHVA_E = (-(h_iHVA_E-infHVAh_E)./tauHVAh_E)*dt + h_iHVA_E; 
 m_iKm_E = (-(m_iKm_E-infKmm_E)./tauKmm_E)*dt + m_iKm_E;
 
 % ION CURRENTS in PY population
 ICL_E = G_lD_E*(VD_E-VCL_E) +gIE_GABA.*(VD_E-VGABA_E);
 IK_E = gg_kl_E.*(VSOMA_E-VK_E) +G_kl_E*(VD_E(i)-VK_E) +G_KCa_E*m_iKCa_E.*m_iKCa_E.*(VD_E-VK_E) +2.9529*G_Km_E*m_iKm_E.*(VD_E-VK_E) +(2.9529*G_Kv_E*m_iKv_E.*(VSOMA_E - VK_E))/200;
  
% ION CONCENTRATIONS %
 Ko_E=(kK_E/F/d_E*(IK_E +Ikpump_E -Ikcc2_E.*(VK_E-VCL_E)./((VK_E-VCL_E)+Vhalf_E)) +Glia_E +D_E./dx_E^2.*(sum(Ko_E(l),2)-4*Ko_E) +D_E_slice./dx_ext_E^2.*(Ko_bath-Ko_E) )*dt + Ko_E;
 % sum(Ko(l),2)-4*Ko                            % GRID DIFFUSION, fast
 % sum(repmat(Ko,1,Ne).*A_diff,1)'-4*Ko         % GRID DIFFUSION, slow (but works)
 % circshift(Ko,1)+circshift(Ko,-1) -2*Ko       % RING DIFFUSION
 Bs_E=(koff_E.*(Bmax_E-Bs_E) -kon_E.*Bs_E.*Ko_E)*dt + Bs_E;     % Glial buffer
 Cli_E=kCL_E/F*(ICL_E +Ikcc2_E.*(VK_E-VCL_E)./((VK_E-VCL_E)+Vhalf_E) )*dt + Cli_E; % Cl
 cai_E=(-5.1819e-5*2.9529*G_HVA_E*m_iHVA_E.^2.*h_iHVA_E.*(VD_E-E_Ca_E)/DCa_E + (0.00024-cai_E)./TauCa_E)*dt + cai_E; %Ca
 
  % SYNAPTIC INPUT
 IE_AMPA=(-IE_AMPA +sigmaE*randn(Ne,1)/sqrt(dt))*dt/tau_AMPA + IE_AMPA;
 % input to the population, rec. E&I + noise + external input
 IE= gEE_AMPA.*(V_AMPA-VD_E) +(gEE_NMDA.*(V_NMDA-VD_E))./(1+Mg/3.57*exp(-0.062*VD_E)) +gIE_GABA.*(VGABA_E-VD_E) +IE_AMPA;
 
 % AMPA 
 gEE_AMPA=(-gEE_AMPA/tau_AMPA + A_E.*sum(S_EE_AMPA(:,fired_E),2) )*dt + gEE_AMPA;
 gEI_AMPA=(-gEI_AMPA/tau_AMPA + A_E.*sum(S_EI_AMPA(:,fired_E),2) )*dt + gEI_AMPA;
 % NMDA
 x_NMDA=(-x_NMDA/tau_NMDA_rise +A_E.*sum(S_EE_NMDA(:,fired_E),2) )*dt + x_NMDA;
 gEE_NMDA=(-gEE_NMDA/tau_NMDA_decay +alpha_NMDA.*x_NMDA.*(1-gEE_NMDA))*dt +gEE_NMDA;
 
 % FIRING PROCESSING
 fired_E=find(VSOMA_E>=0);                          % indices of spikes in the network for one time step
 % Processing of fired_E
 INT_E=intersect(find(VSOMA_E_sp<t_ref),fired_E);   % intersection VSOMA_sp<2 and fired_E
 fired_E=setxor(fired_E,INT_E);                     % remove fired_E elements that intersect with VSOMA_sp<2
 VSOMA_E_sp=VSOMA_E_sp + dt;                        % update time for t* vector
 VSOMA_E_sp(fired_E)=0;
% Record firings
 firings_E=[firings_E; t+0*fired_E,fired_E];
 % fr_E(t)=sum(fired_E)/dt/Ne;                      % E pop rate
 
 % REPRESENTATIVE PATH EXC CELL
 if isempty(Npath)==0
 VEpath(t)=VSOMA_E(Npath(1));
 Kopath(t)=Ko_E(Npath(1));
 Clipath(t)=Cli_E(Npath(1));
 cai_VEpath(t)=cai_E(Npath(1));
 VEpath_IE(t)=gEE_AMPA(Npath(1))*(V_AMPA-VD_E(Npath(1)));
 VEpath_II(t)=gIE_GABA(Npath(1))*(VGABA_E(Npath(1))-VD_E(Npath(1)));
 % clamp
 VEpath_AMPA(t)=gEE_AMPA(Npath(1))*(V_AMPA+60);
 VEpath_GABA(t)=gIE_GABA(Npath(1))*(VGABA_E(Npath(1))-0);
 end
 
 % REPRESENTATIVE NORM EXC CELL
 if isempty(Nnorm)==0
 VEnorm(t)=VSOMA_E(Nnorm(1));
 Konorm(t)=Ko_E(Nnorm(1));
 Clinorm(t)=Cli_E(Nnorm(1));
 cai_VEnorm(t)=cai_E(Nnorm(1));
 VEnorm_IE(t)=gEE_AMPA(Nnorm(1))*(V_AMPA-VD_E(Nnorm(1)));
 VEnorm_II(t)=gIE_GABA(Nnorm(1))*(VGABA_E(Nnorm(1))-VD_E(Nnorm(1)));
 % clamp
 VEnorm_AMPA(t)=gEE_AMPA(Nnorm(1))*(V_AMPA+60);
 VEnorm_GABA(t)=gIE_GABA(Nnorm(1))*(VGABA_E(Nnorm(1))-0);
end
 
%%

%% LFP MODEL
 LFP(t)=k*( sum((VSOMA_E-VD_E)/(kappa_E*S_Dend_E)) );
 % -sum(IE) 
 
%%
 
%% I POPULATOIN INTEGRATION

% synaptic input to each INH cell
II_AMPA=(-II_AMPA +sigmaI*randn(Ni,1)/sqrt(dt))*dt/tau_AMPA + II_AMPA;
 
II= gII_GABA.*(V_GABA-VD_I) +gEI_AMPA.*(V_AMPA-VD_I) +II_AMPA;

% Synaptic conductances
% first order approximations
gII_GABA=(-gII_GABA/tau_GABA + A_I.*sum(S_II_GABA(:,fired_I),2) )*dt + gII_GABA;
gIE_GABA=(-gIE_GABA/tau_GABA + A_I.*sum(S_IE_GABA(:,fired_I),2) )*dt + gIE_GABA;

% K
VK_I=e0_I*log(Ko_E(ind_Ko)/Ki_I);
% NA
VNA_I=e0_I*log(Nao_I/Nai_I); 
% CL
VCL_I=e0_I*log(Cli_I/Clo_I);

% NA-P-PUMP
Ap_I=(1/((1+(Naialpha_I/Nai_I))^3))*(1./((1+(Koalpha_I./Ko_E(ind_Ko))).^2));
Ikpump_I=-2*Imaxsoma_I*Ap_I;
INapump_I=3*Imaxsoma_I*Ap_I;

iDendrite_I= -G_lD_I*(VD_I-VCL_I) -G_kl_I*(VD_I-VK_I) -G_Nal_I*(VD_I-VNA_I) -INapump_I -Ikpump_I + II;

% SOMATIC VOLTAGE
g1_SOMA_I=gg_kl_I +gg_Nal_I +(2.9529*G_Na_I*m_iNa_I.^3.*h_iNa_I) +(2.9529*G_Kv_I*m_iKv_I);
g2_SOMA_I=gg_kl_I*VK_I +gg_Nal_I*VNA_I +(2.9529*G_Na_I*m_iNa_I.^3.*h_iNa_I*VNA_I) +(2.9529*G_Kv_I*m_iKv_I.*VK_I) -INapump_I -Ikpump_I;
VSOMA_I=(VD_I + (kappa_I*S_Soma_I *g2_SOMA_I)) ./ (1+kappa_I*S_Soma_I*g1_SOMA_I);

% POTASSIUM CHANNEL
a_iKv_I=0.02*(VSOMA_I-Vbolz_I)./(1-exp(-(VSOMA_I-Vbolz_I)/9));
b_iKv_I=-0.002*(VSOMA_I-Vbolz_I)./(1-exp((VSOMA_I-Vbolz_I)/9));
tauKvm_I=1./((a_iKv_I+b_iKv_I)*2.9529);
infKvm_I=a_iKv_I./(a_iKv_I+b_iKv_I);

% SODIUM CHANNEL 
am_iNa_I=0.182*(VSOMA_I-10+35)./(1-exp(-(VSOMA_I-10+35)/9));
bm_iNa_I=0.124*(-VSOMA_I+10-35)./(1-exp(-(-VSOMA_I+10-35)/9));
ah_iNa_I=0.024*(VSOMA_I-10+50)./(1-exp(-(VSOMA_I-10+50)/5));
bh_iNa_I=0.0091*(-VSOMA_I+10-75)./(1-exp(-(-VSOMA_I+10-75)/5));
tau_m_I=(1./(am_iNa_I+bm_iNa_I))./2.9529;
tau_h_I=(1./(ah_iNa_I+bh_iNa_I))./2.9529;
m_inf_new_I=am_iNa_I./(am_iNa_I+bm_iNa_I);
h_inf_new_I=1./(1+exp((VSOMA_I-10+65)/6.2));

% VARIABLES INTEGRATION
VD_I=((1/Cm_I)*(iDendrite_I +(VSOMA_I-VD_I)./(kappa_I*S_Dend_I)))*dt + VD_I;
m_iNa_I=(-(m_iNa_I-m_inf_new_I)./tau_m_I)*dt + m_iNa_I;
h_iNa_I=(-(h_iNa_I-h_inf_new_I)./tau_h_I)*dt + h_iNa_I;
m_iKv_I=(-(m_iKv_I-infKvm_I)./tauKvm_I)*dt + m_iKv_I;

% FIRING PROCESSING, INH POP
fired_I=find(VSOMA_I>=0);                        % indices of spikes at one time step
% fr_I(t)=sum(fired_I)/dt/Ni;                      % I pop rate
% Processing of fired_I
INT_I=intersect(find(VSOMA_I_sp<t_ref),fired_I);   % intersection VI_sp<2 and fired_I
fired_I=setxor(fired_I,INT_I);                % remove fired_I elements that intersect with VI_sp<2
VSOMA_I_sp=VSOMA_I_sp + dt;                   % update time for t* vector
VSOMA_I_sp(fired_I)=0;
% Record firings
firings_I=[firings_I; t+0*fired_I,fired_I];

% REPRESENTATIVE INH CELL
VI1(t)=VSOMA_I(1);
VI1_IE(t)=gEI_AMPA(1)*(V_AMPA-VSOMA_I(1));
VI1_II(t)=gII_GABA(1)*(V_GABA-VSOMA_I(1));
% clamp
VI_AMPA(t)=gEI_AMPA(1)*(V_AMPA+60);
VI_GABA(t)=gII_GABA(1)*(V_GABA-0);

%% LFP frequency processing

if t*dt>=sw    
    sw=sw+dsw;              % increase the next sweep
    [freq,psdx,Hz,p_amp] = spect_peak(LFP,dt/1000,50);
         
    if  p_amp>=Tr                     % if peak amplitude > Tr
        maxFR(q,p)=Hz;      % frequency location of the peak
        ampFR(q,p)=p_amp;   % value of the peak        
        break               % stop integration after this frequency is found
    end
    
end


end                    % END of time integration cycle

time=(1:1:t).*dt;      % TIME VECTOR

% save LFP trace
aLFP{q,p}=LFP;
simTime(q,p)=toc;               % amount of simulation time, counter

%% FINAL PLOT
%
figure('units','normalized','outerposition',[0 0 1 1]);

% PLOT
subplot(4,2,3);
plot(firings_I(:,1)*dt,firings_I(:,2),'.','MarkerSize',2,'color','g');
ylabel('Cell index');
axis([0 time(end) 1 Ni]);
set(gca,'FontSize',10);             % set the axis with big font
title('I population');
set(gca,'FontSize',10);             % set the axis with big font
box off;

subplot(4,2,5);
imagesc((reshape(Ko_E,sqrt(Ne),sqrt(Ne)))'); % ,[3 8]
set(gca,'Ydir','normal');
%imagesc((reshape(VK,sqrt(Ne),sqrt(Ne)))');
set(gca,'FontSize',10);             % set the axis with big font
ylabel('Cell index');
xlabel('Cell index');
title('K_{OUT}, mM');
%title('VK, mV');
colormap jet;
colorbar;
box off;

subplot(4,2,7);
imagesc((reshape(Cli_E,sqrt(Ne),sqrt(Ne)))'); % ,[5 25]
set(gca,'Ydir','normal');
ylabel('Cell index');
xlabel('Cell index');
%imagesc((reshape(VGABA,sqrt(Ne),sqrt(Ne)))');
set(gca,'FontSize',10);             % set the axis with big font
title('Cl_{IN}, mM');
%title('VGABA, mV');
colorbar;
box off;

subplot(4,2,4);
plot(time,VI1,'color','g');
set(gca,'FontSize',10);             % set the axis with big font
ylabel('V_I, mV');
xlabel('time, ms');
box off;

if isempty(Npath)==0                        % if there is a pathology

% processing of spiking firings_E
[NP,rep_path]=ismember(firings_E(:,2),Npath);
ind_path=find(rep_path);
[NN,rep_norm]=ismember(firings_E(:,2),Nnorm);
ind_norm=find(rep_norm);

subplot(4,2,1);
plot(firings_E(ind_norm,1)*dt,firings_E(ind_norm,2),'.',firings_E(ind_path,1)*dt,firings_E(ind_path,2),'.','MarkerSize',2);
ylabel('Cell index');
axis([0 time(end) 1 Ne]);
set(gca,'FontSize',10);             % set the axis with big font
title('E population');
%legend('KCC2(+)','KCC2(-)');
set(gca,'FontSize',10);             % set the axis with big font
box off;

subplot(4,2,2);
plot(time,VEnorm,time,VEpath);
set(gca,'FontSize',10);             % set the axis with big font
ylabel('V_E, mV');
title(sprintf('%d S_{EE AMPA} %d S_{IE GABA}',(q-1)*dq,(p-1)*dp));
%legend('KCC2(+)','KCC2(-)');
box off;

subplot(4,2,6);
plot(time,Konorm,time,Kopath);
set(gca,'FontSize',10);             % set the axis with big font
ylabel('K_{OUT}, mM');
%legend('KCC2(+)','KCC2(-)');
box off;

subplot(4,2,8);
plot(time,Clinorm,time,Clipath);
set(gca,'FontSize',10);             % set the axis with big font
ylabel('Cl_{IN}, mM');
%legend('KCC2(+)','KCC2(-)');
box off;

else                                % if there is no pathology
    
subplot(4,2,1);
plot(firings_E(:,1)*dt,firings_E(:,2),'.','MarkerSize',2,'color','b');
ylabel('Cell index');
axis([0 time(end) 1 Ne]);
set(gca,'FontSize',10);             % set the axis with big font
title('E population');
set(gca,'FontSize',10);             % set the axis with big font
box off;
    
subplot(4,2,2);
plot(time,VEnorm,'color','b');
title(sprintf('%d S_{EE AMPA} %d S_{IE GABA}',(q-1)*dq,(p-1)*dp));
set(gca,'FontSize',10);             % set the axis with big font
ylabel('V_E, mV');
box off;

subplot(4,2,6);
plot(time,Konorm);
set(gca,'FontSize',10);             % set the axis with big font
ylabel('Ko, mM');
box off;

subplot(4,2,8);
plot(time,Clinorm,'blue');
set(gca,'FontSize',10);             % set the axis with big font
ylabel('Cli, mM');
box off;

end
%}
%%

saveas(gcf,sprintf('Net_SYNPAR_%d_%d.jpg',(q-1)*dq,(p-1)*dp),'jpg');  % save *.jpg
% parsave(sprintf('Net_SYNPAR_%d_%d.mat',(q-1)*dq,(p-1)*dp),(q-1)*dq,(p-1)*dp,Npath,Nnorm,time,simTime(q,p),dt,Ne,Ni,firings_E,firings_I,Konorm,Kopath,Ko_E,Clinorm,Clipath,Cli_E,VEnorm,VI1,VEpath,LFP);           % save *.mat

end

end

close all

save('SYNPAR.mat','simTime','aLFP','maxFR','ampFR','dt','P_axis','Q_axis');           % save *.mat

%%

figure('units','normalized','outerposition',[0 0 0.5 0.5]);

colormap parula;
imagesc(Q_axis,P_axis,maxFR, [0 5])
set(gca,'Ydir','normal');
%set(gca,'Xdir','normal');
set(gca,'Fontsize',30);
ylabel('g_{EE AMPA} (%)');
xlabel('g_{IE GABA} (%)');
title('Frequency (Hz)')
colorbar

%%

saveas(gcf,'Frequency.jpg','jpg');  % save *.jpg

clear;