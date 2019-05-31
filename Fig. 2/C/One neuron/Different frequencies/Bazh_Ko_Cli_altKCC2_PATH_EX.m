%% NUMERICAL PARAMETERS
tic

T=60000;        % total time, mS
Tst=5000;      % duration of stimulation, ms
dt=0.05;         % time step, ms
t=(0:1:round(T/dt))*dt;

%% Stimulus parameters
ts=0;
Hz_max=11;  % max stimulation intensity (-1 in fact)
dHz=1;      % step of stimulation intensity
Tseizure=0;

%% PY parameters
% Mg
Mg=1;
% CL
Clo_E=130;        % mM
Vhalf_E=40;       % KCC2 1/2
Ikcc2_E=0;        % KCC2 max current
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
% KCC2 norm
HCO3o=26;       % mM
HCO3i=16;       % mM


%% SN border, to be CALCULATED!!!
a=404.5;
b=-0.8712;
c=27.9;
d=-0.1488;

 

%% loop over trials
parfor l=1:1:round(Hz_max/dHz)  % loop over stimulation intensities, should be PARFOR
    
    tt=0;          % time inside seizure count    

if l-1==0
    ts=0;
else    
ts=round(1000/((l-1)*dHz));
end

dts=ts;                 % next stimulation interval, ts+dts

HZ(l)=(l-1)*dHz;              % frequency of stimulation, Hz

%% ICS/Parameters Synaptic input, approximations from Chizhov et al 2002

% SYNAPTIC INPUT
% GABA
ggGABA_ext=0;
gGABA_ext=0;
% AMPA
ggAMPA_ext=0;
gAMPA_ext=0;
% NMDA
ggNMDA_ext=0;
gNMDA_ext=0;

% GABA-A
alpha1_GABA=0.1;          % kHz
alpha2_GABA=0.1;          % kHz
%AMPA
alpha1_AMPA=0.5;          % kHz
alpha2_AMPA=0.5;          % kHz
V_AMPA=5;                 % mV
%NMDA
alpha1_NMDA=0.05;         % kHz
alpha2_NMDA=0.05;         % kHz 
VNMDA=10;                 % mV

gGABA_max=3;          % mS/cm^2, estimated from Chizhov 2
gAMPA_max=1;          % mS/cm^2, estimated from Chizhov 3
gNMDA_max=2;          % mS/cm^2


% INITIAL CONDITIONS (rest state, KCC2(-))
Ko=3.46;             % mM
Cli=11.30;            % mM
cai=0.00;            % mM
Bs=499.92;           % mM
VD=-65.37;           % mV
VSOMA=-65.45;        % mV
VGABA=-70;           % mV
m_iKv=0.00;       % 1
m_iNa=0.01;       % 1
h_iNa=0.83;       % 1
m_iKm=0.01;       % 1
m_iNaD=0.01;      % 1
h_iNaD=0.84;      % 1
m_iNapD=0.00;     % 1
m_iKCa=0.00;      % 1
m_iHVA=0.00;      % 1
h_iHVA=0.58;      % 1


%% loop over time
for i=1:1:round(T/dt)
  
    % SN border approximation
     Cl_appr=a*exp(b*Ko(i)) + c*exp(d*Ko(i));
     % Cl_appr=a1*exp(-((Ko(i)-b1)/c1)^2) +a2*exp(-((Ko(i)-b2)/c2)^2) +a3*exp(-((Ko(i)-b3)/c3)^2) +a4*exp(-((Ko(i)-b4)/c4)^2) +a5*exp(-((Ko(i)-b5)/c5)^2) +a6*exp(-((Ko(i)-b6)/c6)^2);
        
     if Cli(i)>Cl_appr       % more than SN     
        tt=tt+1;     
     end   
   
% delta function approximation
    if i*dt<=Tst       % stimulation only during the stimulation period
 if i*dt==ts        % generation of stimulus times
    delta_I=1/dt;
    delta_E=1/dt;
    ts=ts+dts;
 else
     delta_I=0;
     delta_E=0;
 end
 else               % count the seizure duration after the stimulus
     ts=0;
     delta_I=0;
     delta_E=0;          
    end
 
 % GABA-A ext
 gGABA_ext(i+1)=ggGABA_ext(i)*dt + gGABA_ext(i);
 ggGABA_ext(i+1)=(alpha1_GABA.*alpha2_GABA.*( delta_I.*(1-gGABA_ext(i))./K(alpha1_GABA,alpha2_GABA)-gGABA_ext(i)-(1/alpha1_GABA +1/alpha2_GABA).*ggGABA_ext(i))).*dt +ggGABA_ext(i);
 
% AMPA ext
 gAMPA_ext(i+1)=ggAMPA_ext(i)*dt + gAMPA_ext(i);
 ggAMPA_ext(i+1)=(alpha1_AMPA.*alpha2_AMPA.*( delta_E.*(1-gAMPA_ext(i))./K(alpha1_AMPA,alpha2_AMPA)-gAMPA_ext(i)-(1/alpha1_AMPA +1/alpha2_AMPA).*ggAMPA_ext(i))).*dt + ggAMPA_ext(i);        
 % ALGEBRAIC EQUATIONS
 
 % NMDA ext
 gNMDA_ext(i+1)=ggNMDA_ext(i)*dt + gNMDA_ext(i);
 ggNMDA_ext(i+1)=(alpha1_NMDA.*alpha2_NMDA.*( delta_E.*(1-gNMDA_ext(i))./K(alpha1_NMDA,alpha2_NMDA)-gNMDA_ext(i)-(1/alpha1_NMDA +1/alpha2_NMDA).*ggNMDA_ext(i))).*dt +ggNMDA_ext(i);
 
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
 VGABA(i)=e0_E*log((4*Cli(i)+HCO3i)./(4*Clo_E+HCO3o)); 
 
 % dendrite current
 f_NMDA=1/(1+Mg/3.57*exp(-0.062*VD(i)));
 iDendrite= -gNMDA_max*gNMDA_ext(i)*f_NMDA*(VD(i)-VNMDA) -gGABA_max*gGABA_ext(i)*(VD(i)-VGABA(i)) -gAMPA_max.*gAMPA_ext(i)*(VD(i)-V_AMPA) -G_lD_E*(VD(i)-VCL) -G_kl_E*(VD(i)-VKe) -G_Nal_E*(VD(i)-VNAe) -2.9529*G_NaD_E*m_iNaD(i)^3*h_iNaD(i)*(VD(i)-VNAe) -G_NapD_E*m_iNapD(i)*(VD(i)-VNAe) -G_KCa_E*m_iKCa(i)^2*(VD(i) - VKe) -2.9529*G_Km_E*m_iKm(i)*(VD(i) - VKe) -2.9529*G_HVA_E*m_iHVA(i)^2*h_iHVA(i)*(VD(i)-E_Ca_E) -INapump -Ikpump;
 
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
ICL = G_lD_E*(VD(i)-VCL) +gGABA_max*gGABA_ext(i)*(VD(i)-VGABA(i));
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
Ko(i+1)=(kK_E/F/d_E*(IK +Ikpump +Ikpump -Ikcc2_E*(VKe-VCL)/((VKe-VCL)+Vhalf_E) ) +Glia )*dt + Ko(i);
Bs(i+1)=(koff_E*(Bmax_E-Bs(i)) -kon*Bs(i)*Ko(i))*dt + Bs(i);
Cli(i+1)=kCL_E/F*(ICL +Ikcc2_E*(VKe-VCL)/((VKe-VCL)+Vhalf_E) )*dt + Cli(i);
cai(i+1)=(-5.1819e-5* 2.9529*G_HVA_E*m_iHVA(i)^2*h_iHVA(i) * (VD(i) - E_Ca_E)/DCa_E + (0.00024-cai(i))/TauCa_E)*dt + cai(i);     
 
end

% seizure time
Tseizure(l)=tt*dt;


%% PLOT
%{
figure;

subplot(3,1,1);
plot(t(1:end-1),VSOMA);
set(gca,'FontSize',20);             % set the axis with big font
title(sprintf('Bazh NORM KCC2'));
xlabel('time, ms');
ylabel('V_{S}, mV');

subplot(3,1,2);
plot(t,Cli);
set(gca,'FontSize',20);             % set the axis with big font
xlabel('time, ms');
ylabel('Cli, mM');

subplot(3,1,3);
plot(t,Ko);
set(gca,'FontSize',20);             % set the axis with big font
xlabel('time, ms');
ylabel('Ko, mM');
%}
%%


end

VD_rest(1:1:length(HZ))=-64.70;
% calculated from ICs
figure;

%% VGABA-intensity plot
plot(HZ,Tseizure,'blue','LineWidth',6);
axis([0 max(HZ) 0 T])
set(gca,'FontSize',30);
xlabel('Intensity of stimulation, Hz');
ylabel('Afterdischarge duration, ms');
box off;
title('KCC2(-)')

toc
%%