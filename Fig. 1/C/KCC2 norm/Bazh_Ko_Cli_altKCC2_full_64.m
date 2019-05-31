% NUMERICAL PARAMETERS
%%
T=500;        % total time, mS
dt=0.05;         % time step, ms

% MODEL PARAMETERS

% KCC2 norm
I=0.29;            % external input for VSOMA ~ -64 mV

HCO3o=26;       % mM
HCO3i=16;       % mM

% Synaptic input
% GABA-A
alpha1_GABA=5;
alpha2_GABA=0.120;
gImax_ext=0.062;             % mS/cm^2


%% PY parameters
% CL
Clo_E=130;        % mM
Vhalf_E=40;       % KCC2 1/2
Ikcc2_E=2;        % KCC2 max current
kCL_E=100;        % CL conversion factor, 1000 cm^3
% K
Ki_E=150;         % intra K, mM
kK_E=10;          % K conversion factor, 1000 cm^3
d_E=0.15;         % ratio Volume/surface, mu m
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
% constants
F=96489;        % coul/M
% Conductances
% somatic
G_Na_E=3450.0;     % Na, mS/cm^2
G_Kv_E=200.0;      % Kv-channel mS/cm^2
gg_kl_E=0.042;     % K leak, mS/cm^2
gg_Nal_E=0.0198;   % Na leak, mS/cm^2
Vbolz_E=22;        % mV
% dendritic
G_NaD_E=1.1;       % mS/cm^2
G_NapD_E=3.5;      % mS/cm^2
G_HVA_E=0.0195;    % mS/cm^2
G_kl_E=0.044;      % K leak, mS/cm^2
G_lD_E=0.01;       % Clleak, mS/cm^2
G_Nal_E=0.02;      % Na leak, mS/cm^2
E_Ca_E=140;        % mV
TauCa_E=800;       % ms
DCa_E=0.85;        % ???
G_KCa_E=2.5;       % mS/cm^2
G_Km_E=0.01;       % mS/cm^2
% Pump parametes and glial buffer
Koalpha_E=3.5;    % mM
Naialpha_E=20;    % mM
Imaxsoma_E=25;    % muA/cm^2
Imaxdend_E=25;    % muA/cm^2
Kothsoma_E=15;    % mM
koff_E=0.0008;    % 1/ mM / ms
K1n_E=1.0;        % 1/ mM
Bmax_E=500;       % mM


% I=0;            % external input for VSOMA ~ -64 mV
%%
% INITIAL CONDITIONS (rest state, KCC2(+))
Ko(1)=4;             % mM
Cli(1)=4.06;            % mM
cai(1)=0.00;            % mM
Bs(1)=499.85;           % mM
VD(1)=-68.42;           % mV
VSOMA(1)=-68.47;        % mV
m_iKv(1)=0.00;       % 1
m_iNa(1)=0.01;       % 1
h_iNa(1)=0.89;       % 1
m_iKm(1)=0.01;       % 1
m_iNaD(1)=0.01;      % 1
h_iNaD(1)=0.89;      % 1
m_iNapD(1)=0.00;     % 1
m_iKCa(1)=0.00;      % 1
m_iHVA(1)=0.00;      % 1
h_iHVA(1)=0.62;      % 1

ggI_ext=0;
gI_ext=0;
ts=200;            % ms, time of stimuli applicaiton



%%
for i=1:1:round(T/dt)
  
 % ALGEBRAIC EQUATIONS
    
 % Na-P-pump
 Ap=(1/((1+(Koalpha_E/Ko(i)))*(1+(Koalpha_E/Ko(i)))))*(1/((1+(Naialpha_E/Nai_E))*(1+(Naialpha_E/Nai_E))*(1+(Naialpha_E/Nai_E))));
 Ikpump=-2*Imaxsoma_E*Ap;
 INapump=3*Imaxsoma_E*Ap;
 
 % reversal potentials on soma and dendrite 
 % K
 VKe=e0_E*log(Ko(i)/Ki_E);
 % NA
 VNAe=e0_E*log(Nao_E/Nai_E); 
 % CL
 VCL=e0_E*log(Cli(i)/Clo_E);
 % VGABA
 VGABA=e0_E*log((4*Cli(i)+HCO3i)./(4*Clo_E+HCO3o));
 
 if i*dt==ts     
    delta_I=1/dt;
 else
     delta_I=0;
 end
 
 % external input to the neuron
 gI_ext(i+1)=ggI_ext(i)*dt + gI_ext(i);
 ggI_ext(i+1)=(alpha1_GABA.*alpha2_GABA.*( delta_I./K(alpha1_GABA,alpha2_GABA)-gI_ext(i)-(1/alpha1_GABA +1/alpha2_GABA).*ggI_ext(i))).*dt +ggI_ext(i);
 
 % dendrite current
 iDendrite=I -gImax_ext*gI_ext(i)*(VD(i)-VGABA) -G_lD_E*(VD(i)-VCL) -G_kl_E*(VD(i)-VKe) -G_Nal_E*(VD(i)-VNAe) -2.9529*G_NaD_E*m_iNaD(i)^3*h_iNaD(i)*(VD(i)-VNAe) -G_NapD_E*m_iNapD(i)*(VD(i)-VNAe) -G_KCa_E*m_iKCa(i)^2*(VD(i) - VKe) -2.9529*G_Km_E*m_iKm(i)*(VD(i) - VKe) -2.9529*G_HVA_E*m_iHVA(i)^2*h_iHVA(i)*(VD(i)-E_Ca_E) -INapump -Ikpump;
 
 % somatic voltage
 g1_SOMA=gg_kl_E +gg_Nal_E +(2.9529*G_Na_E*m_iNa(i)^3*h_iNa(i)) +(2.9529*G_Kv_E*m_iKv(i));
 g2_SOMA=gg_kl_E*VKe +gg_Nal_E*VNAe +(2.9529*G_Na_E*m_iNa(i)^3*h_iNa(i)*VNAe) +(2.9529*G_Kv_E*m_iKv(i)*VKe) -INapump -Ikpump;

 VSOMA(i)=(VD(i) + (kappa_E*S_Soma_E *g2_SOMA)) / (1+kappa_E*S_Soma_E*g1_SOMA);
 
 % Ikv, POTASSIUM CHANNEL
a_iKv=0.02*(VSOMA(i)-Vbolz_E)/(1-exp(-(VSOMA(i)-Vbolz_E)/9));
b_iKv=-0.002*(VSOMA(i)-Vbolz_E)/(1-exp((VSOMA(i)-Vbolz_E)/9));
tauKvm=1/((a_iKv+b_iKv)*2.9529);
infKvm=a_iKv/(a_iKv+b_iKv);

 % INA, SODIUM CHANNEL 
am_iNa=0.182*(VSOMA(i)-10+35)/(1-exp(-(VSOMA(i)-10+35)/9));
bm_iNa=0.124*(-VSOMA(i)+10-35)/(1-exp(-(-VSOMA(i)+10-35)/9));
ah_iNa=0.024*(VSOMA(i)-10+50)/(1-exp(-(VSOMA(i)-10+50)/5));
bh_iNa=0.0091*(-VSOMA(i)+10-75)/(1-exp(-(-VSOMA(i)+10-75)/5));
tau_m=(1/(am_iNa+bm_iNa))/2.9529;
tau_h=(1/(ah_iNa+bh_iNa))/2.9529;
m_inf_new=am_iNa/(am_iNa+bm_iNa);
h_inf_new=1/(1+exp((VSOMA(i)-10+65)/6.2));

% NaP, D current
minfiNapD = 0.02/(1 + exp(-(VD(i)+42)/5));

% INa D, sodium channel
am_iNaD=0.182*(VD(i)-10+35)/(1-exp(-(VD(i)-10+35)/9));
bm_iNaD=0.124*(-VD(i)+10-35)/(1-exp(-(-VD(i)+10-35)/9));
ah_iNaD=0.024*(VD(i)-10+50)/(1-exp(-(VD(i)-10+50)/5));
bh_iNaD=0.0091*(-VD(i)+10-75)/(1-exp(-(-VD(i)+10-75)/5));
minf_newD = am_iNaD/(am_iNaD+bm_iNaD);
hinf_newD = 1/(1+exp((VD(i)-10+65)/6.2));
tau_mD = (1/(am_iNaD+bm_iNaD))/2.9529;
tau_hD = (1/(ah_iNaD+bh_iNaD))/2.9529;

%%%% iKCa %%%%
minf_iKCa = (48*cai(i)*cai(i)/0.03)/(48*cai(i)*cai(i)/0.03 + 1);
taum_iKCa = (1/(0.03*(48*cai(i)*cai(i)/0.03 + 1)))/4.6555;

%%%% IHVA %%%%
am_iHVA = 0.055*(-27 - VD(i))/(exp((-27-VD(i))/3.8) - 1);
bm_iHVA = 0.94*exp((-75-VD(i))/17);
ah_iHVA = 0.000457*exp((-13-VD(i))/50);
bh_iHVA = 0.0065/(exp((-VD(i)-15)/28) + 1);
tauHVAh = 1/((ah_iHVA+bh_iHVA)*2.9529);
infHVAh = ah_iHVA/(ah_iHVA+bh_iHVA);
tauHVAm = 1/((am_iHVA+bm_iHVA)*2.9529);
infHVAm = am_iHVA/(am_iHVA+bm_iHVA);

%%% IKM %%%%
 am_iKm = 0.001 * (VD(i) + 30) / (1 - exp(-(VD(i) + 30)/9));
 bm_iKm = -0.001 * (VD(i) + 30) / (1 - exp((VD(i) + 30)/9));
 tauKmm = 1/((am_iKm+bm_iKm)*2.9529);
 infKmm = am_iKm/(am_iKm+bm_iKm);

% ION CURRENTS
ICL = G_lD_E*(VD(i)-VCL);
IK = gg_kl_E*(VSOMA(i)-VKe) +G_kl_E*(VD(i)-VKe) +G_KCa_E*m_iKCa(i)*m_iKCa(i)*(VD(i)-VKe) +2.9529*G_Km_E*m_iKm(i)*(VD(i)-VKe) +(2.9529*G_Kv_E*m_iKv(i)*(VSOMA(i)-VKe))/200;

% GLIA
kon=koff_E/(1+exp((Ko(i)-Kothsoma_E)/(-1.15)));
Glia=koff_E*(Bmax_E-Bs(i))/K1n_E -kon/K1n_E*Bs(i)*Ko(i);

%INTEGRATION
VD(i+1) = ((1/Cm_E)*(iDendrite +(VSOMA(i)-VD(i)) / (kappa_E*S_Dend_E)))*dt + VD(i);
m_iNa(i+1) =(-(m_iNa(i)-m_inf_new)/tau_m)*dt + m_iNa(i);
h_iNa(i+1) =(-(h_iNa(i)-h_inf_new)/tau_h)*dt + h_iNa(i);
m_iKv(i+1) =(-(m_iKv(i)-infKvm)/tauKvm)*dt + m_iKv(i);
m_iNaD(i+1) =(-(m_iNaD(i) - minf_newD)/tau_mD)*dt +m_iNaD(i);
h_iNaD(i+1) =(-(h_iNaD(i) - hinf_newD)/tau_hD)*dt +h_iNaD(i);
m_iNapD(i+1)=(-(m_iNapD(i) - minfiNapD)/0.1992)*dt +m_iNapD(i);
m_iKCa(i+1) =(-(1/taum_iKCa)*(m_iKCa(i) - minf_iKCa))*dt + m_iKCa(i);
m_iHVA(i+1) = (-(m_iHVA(i)-infHVAm)/tauHVAm)*dt + m_iHVA(i);
h_iHVA(i+1) = (-(h_iHVA(i)-infHVAh)/tauHVAh)*dt + h_iHVA(i);
m_iKm(i+1) = (-(m_iKm(i)-infKmm)/tauKmm)*dt + m_iKm(i);

% ION CONCENTRATION
%Ko(i+1)=(kK_E/F/d_E*(IK +Ikpump +Ikpump -Ikcc2_E*(VKe-VCL)/((VKe-VCL)+Vhalf_E) ) +Glia )*dt + Ko(i);
Ko(i+1)=4;
Bs(i+1)=(koff_E*(Bmax_E-Bs(i)) -kon*Bs(i)*Ko(i))*dt + Bs(i);
Cli(i+1)=kCL_E/F*(ICL +Ikcc2_E*(VKe-VCL)/((VKe-VCL)+Vhalf_E) )*dt + Cli(i);
%Cli(i+1)=9;
cai(i+1)=(-5.1819e-5* 2.9529*G_HVA_E*m_iHVA(i)^2*h_iHVA(i) * (VD(i) - E_Ca_E)/DCa_E + (0.00024-cai(i))/TauCa_E)*dt + cai(i);

end

t=(1:1:round(T/dt))*dt;
%%

%%
subplot(2,1,1);
plot(t,VSOMA);
set(gca,'FontSize',30);             % set the axis with big font
xlabel('time, ms');
ylabel('V_{SOMA}, mV');
box off;

subplot(2,1,2)
plot(t,gI_ext(1:end-1));
%axis([3.5 7 3.5 10]);
set(gca,'FontSize',30);             % set the axis with big font
xlabel('time, ms');
ylabel('g_{ext}, mS/cm^2');
%%

%VSOMA(end)

min(VSOMA(100/dt:end))-VSOMA(end)
%VSOMA(end)-max(VSOMA(200/dt:end))
