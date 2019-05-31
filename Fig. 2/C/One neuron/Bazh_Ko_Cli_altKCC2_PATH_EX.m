% NUMERICAL PARAMETERS
T=30000;        % total time, mS
Tst=3000;      % duration of stimulation, ms
dt=0.05;         % time step, ms

% MODEL PARAMETERS

% CL
Clo=130;        % mM

Vhalf=40;       % KCC2 1/2
Ikcc2=0;      % KCC2 max current 0.5
                % 0.01 - norm               
                % 0.005 - path
                
HCO3o=26;       % mM
HCO3i=16;       % mM
                                
kCL=100;        % CL conversion factor, 1000 cm^3

% K
Ki=129.34;         % intra K, mM
kK=10;          % K conversion factor, 1000 cm^3
Vbolz=22.8;     % constant for K delayed-rect current
d=0.15;         % ratio Volume/surface, mu m

% NA
Nao=125.90;        % extra Na, mM
Nai=20.61;         % intra Na, mM
kNa=10;         % NA conversion, 1000 cm^3


% single cell and membrane parameters
Cm=0.75;         % mu F/cm^2
e0=26.6393;      % kT/F, in Nernst equation
kappa=10000;     % conductance between compartments, Ohm
S_Soma=0.000001; % cm^2
S_Dend=0.000165; % cm^2


% constants
F=96489;        % coul/M

% Conductances

% active
G_Na=3450.0;     % Na, mS/cm^2
G_Kv=200.0;      % Kv-channel mS/cm^2

% dendrite leaks
G_kl=0.035;      % mS/cm^2
G_lD=0.01;       % Cl, mS/cm^2
G_Nal=0.02;      % mS/cm^2

% somatic leaks
gg_kl=0.042;     % mS/cm^2
gg_Nal=0.022;   % mS/cm^2

% Pump parametes and glial buffer

Koalpha=2.5;    % mM
Naialpha=20;    % mM
Imaxsoma=20;    % muA/cm^2
Imaxdend=20;    % muA/cm^2
Kothsoma=10;    % mM
koff=0.0008;    % 1/ mM / ms
K1n=1.0;        % 1/ mM
Bmax=500;       % mM

Imean=0;    % 0.14 muA/cm^2
sigma=0;


ts=0;

Hz_max=7;  % max stimulation intensity
dHz=1;      % step of stimulation intensity

Tseizure=0;

t=(0:1:round(T/dt))*dt;

% Area of excitability, SN BORDER APPR. PARAMETERS
 a1 = 35.57;
 b1 = 1.498;
 c1 = 0.299;
 a2 = 5.352;
 b2 = 1.91;
 c2 = 0.2913;
 a3 = 50.37;
 b3 = 0.8663;
 c3 = 1.39;
 a4 = -0.3178;
 b4 = 2.624;
 c4 = 0.1226;
 a5 = -20.68;
 b5 = -0.009649;
 c5 = 8.127;
 a6 = 4.162e+14;
 b6 = -283.8;
 c6 = 52.51;
 

%% loop over trials
for l=1:1:round(Hz_max/dHz)  % loop over stimulation intensities, should be PARFOR
    
    tt=0;          % time inside seizure count
    
% Synaptic input
% GABA-A
alpha1_GABA=5;
alpha2_GABA=0.120;

%AMPA
alpha1_AMPA=0.185;
alpha2_AMPA=0.185;
V_AMPA=0;                   % mV

if l-1==0
    ts=0;
else    
ts=round(1000/((l-1)*dHz));
end

dts=ts;                 % next stimulation interval, ts+dts

HZ(l)=(l-1)*dHz;              % frequency of stimulation, Hz
gImax_ext=3;          % mS/cm^2, estimated from Chizhov 
gEmax_ext=2;          % mS/cm^2, estimated from Chizhov

% INITIAL CONDITIONS rest state (Ikcc2=0.5)

Ko=3.25;      % mM
Cli=13.33;     % mM
VGABA=e0*log((4*Cli+HCO3i)./(4*Clo+HCO3o));

Bs=495.43;      % mM
VD=-60.65;      % mV
VSOMA=VD;    % mV
m_iKv=0.000;    % 1
m_iNa=0.027;    % 1
h_iNa=0.714;    % 1

% external input
ggI_ext=0;
gI_ext=0;
ggE_ext=0;
gE_ext=0;

%% loop over time
for i=1:1:round(T/dt)
  
 % ALGEBRAIC EQUATIONS
    
 % Na-P-pump
 Ap=(1/((1+(Koalpha/Ko(i)))*(1+(Koalpha/Ko(i)))))*(1/((1+(Naialpha/Nai))*(1+(Naialpha/Nai))*(1+(Naialpha/Nai))));
 Ikpump=-2*Imaxsoma*Ap;
 INapump=3*Imaxsoma*Ap;
 
 % reversal potentials on soma and dendrite
 % K
 VK=e0*log(Ko(i)/Ki);
 % NA
 VNA=e0*log(Nao/Nai);
 % CL
 VCL=e0*log(Cli(i)/Clo);
 % VGABA
 VGABA(i)=e0*log((4*Cli(i)+HCO3i)./(4*Clo+HCO3o));
 
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
 
 % SN border approximation
     Cl_appr=a1*exp(-((Ko(i)-b1)/c1)^2) +a2*exp(-((Ko(i)-b2)/c2)^2) +a3*exp(-((Ko(i)-b3)/c3)^2) +a4*exp(-((Ko(i)-b4)/c4)^2) +a5*exp(-((Ko(i)-b5)/c5)^2) +a6*exp(-((Ko(i)-b6)/c6)^2);
        
     if Cli(i)>Cl_appr       % more than SN     
        tt=tt+1;     
    end   
     
 % GABA-A ext
 gI_ext(i+1)=ggI_ext(i)*dt + gI_ext(i);
 ggI_ext(i+1)=(alpha1_GABA.*alpha2_GABA.*( delta_I.*(1-gI_ext(i))./K(alpha1_GABA,alpha2_GABA)-gI_ext(i)-(1/alpha1_GABA +1/alpha2_GABA).*ggI_ext(i))).*dt +ggI_ext(i);
 
 % AMPA ext
 gE_ext(i+1)=ggE_ext(i)*dt + gE_ext(i);
 ggE_ext(i+1)=(alpha1_AMPA.*alpha2_AMPA.*( delta_E.*(1-gE_ext(i))./K(alpha1_AMPA,alpha2_AMPA)-gE_ext(i)-(1/alpha1_AMPA +1/alpha2_AMPA).*ggE_ext(i))).*dt + ggE_ext(i);

 
 % dendrite current
 iDendrite= -G_lD*(VD(i)-VCL) -G_kl*(VD(i)-VK) -G_Nal*(VD(i)-VNA) -INapump -Ikpump +Imean +sigma*randn(1,1) -gImax_ext*gI_ext(i)*(VD(i)-VGABA(i)) -gEmax_ext.*gE_ext(i)*(VD(i)-V_AMPA);
 
 % somatic voltage
 g1_SOMA=gg_kl +gg_Nal +(2.9529*G_Na*m_iNa(i)^3*h_iNa(i)) +(2.9529*G_Kv*m_iKv(i));
 g2_SOMA=gg_kl*VK +gg_Nal*VNA +(2.9529*G_Na*m_iNa(i)^3*h_iNa(i)*VNA) +(2.9529*G_Kv*m_iKv(i)*VK) -INapump -Ikpump;
 
 % POTASSIUM CHANNEL
a_iKv=0.02*(VSOMA(i)-Vbolz)/(1-exp(-(VSOMA(i)-Vbolz)/9));
b_iKv=-0.002*(VSOMA(i)-Vbolz)/(1-exp((VSOMA(i)-Vbolz)/9));
tauKvm=1/((a_iKv+b_iKv)*2.9529);
infKvm=a_iKv/(a_iKv+b_iKv);

 % SODIUM CHANNEL
am_iNa=0.182*(VSOMA(i)-10+35)/(1-exp(-(VSOMA(i)-10+35)/9));
bm_iNa=0.124*(-VSOMA(i)+10-35)/(1-exp(-(-VSOMA(i)+10-35)/9));
ah_iNa=0.024*(VSOMA(i)-10+50)/(1-exp(-(VSOMA(i)-10+50)/5));
bh_iNa=0.0091*(-VSOMA(i)+10-75)/(1-exp(-(-VSOMA(i)+10-75)/5));

tau_m=(1/(am_iNa+bm_iNa))/2.9529;
tau_h=(1/(ah_iNa+bh_iNa))/2.9529;
m_inf_new=am_iNa/(am_iNa+bm_iNa);
h_inf_new=1/(1+exp((VSOMA(i)-10+65)/6.2));

% ION CURRENTS
SigCl = G_lD*(VD(i)-VCL) +gImax_ext*gI_ext(i)*(VD(i)-VGABA(i));
SigIkints = gg_kl*(VSOMA(i)-VK) + (2.9529*G_Kv*m_iKv(i)*(VSOMA(i) - VK))/200;

% GLIA
kon=koff/(1+exp((Ko(i)-Kothsoma)/(-1.15)));
Glia=koff*(Bmax-Bs(i))/K1n -kon/K1n*Bs(i)*Ko(i);

%INTEGRATION
VD(i+1)=((1/Cm)*(iDendrite +(VSOMA(i)-VD(i)) / (kappa*S_Dend)))*dt + VD(i);
VSOMA(i+1)=(VD(i) + (kappa*S_Soma *g2_SOMA)) / (1+kappa*S_Soma*g1_SOMA);
m_iNa(i+1)=(-(m_iNa(i)-m_inf_new)/tau_m)*dt + m_iNa(i);
h_iNa(i+1)=(-(h_iNa(i)-h_inf_new)/tau_h)*dt + h_iNa(i);
m_iKv(i+1)=(-(m_iKv(i)-infKvm)/tauKvm)*dt + m_iKv(i);

Ko(i+1)=(kK/F/d*(SigIkints +Ikpump -Ikcc2*(VK-VCL)/((VK-VCL)+Vhalf) )  +Glia)*dt + Ko(i);

Bs(i+1)=(koff*(Bmax-Bs(i)) -kon*Bs(i)*Ko(i))*dt + Bs(i);
Cli(i+1)=kCL/F*(SigCl +Ikcc2*(VK-VCL)/((VK-VCL)+Vhalf) )*dt + Cli(i);


end

% average of the last 100ms
Tseizure(l)=tt*dt;

%Cli_mean(l)=mean(Cli(round(500)/dt:end));
%Ko_mean(l)=mean(Ko(round(500)/dt:end));
%VGABA_mean(l)=mean(VGABA(round(500)/dt:end));


% KO(1:round(T/dt)+1)=Ko_mean(l);
% CLI(1:round(T/dt)+1)=Cli_mean(l);


%% PLOT

figure;

subplot(3,1,1);
plot(t,VSOMA);
set(gca,'FontSize',20);             % set the axis with big font
title(sprintf('Bazh PATH KCC2',sigma));
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

%%


end

VD_rest(1:1:length(HZ))=-64.70;
% calculated from ICs
figure;

%% VGABA-intensity plot
plot(HZ,Tseizure,'blue','LineWidth',6);
% axis([0 max(HZ) 0 1000])
set(gca,'FontSize',60);
xlabel('Intensity of stimulation, Hz');
ylabel('Duration of spiking, ms');
title('KCC2 norm, g_{GABA}=3 uS/cm^2, g_{AMPA}=2 uS/cm^2')
%%
