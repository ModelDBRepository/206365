%% NUMERICAL PARAMETERS

% load the SN border
load('Ko_Cli.mat','Cli1','Cli2','Ko1','Ko2');

T=120000;        % total time, mS
Tst=5000;      % duration of stimulation, ms
dt=0.05;         % time step, ms
t=(0:1:round(T/dt))*dt;

%% Stimulus parameters
ts=0;
Hz_max=11;  % max stimulation intensity (-1)
dHz=1;      % step of stimulation intensity
Tseizure=0;

%% Network parameters
Ne=841; Ni=225; 
% Mg
Mg=1;          % mM, Mg concentration for NMDA synapses
% CL
Clo=130;        % mM                                
Vhalf=40;                % KCC2 1/2
HCO3o=26;       % mM
HCO3i=16;       % mM
kCL=100;        % CL conversion factor, 1000 cm^3
% K
Ki_e=150;         % intra K, mM
kK=10;             % K conversion factor, 1000 cm^3
d=0.15;            % ratio Volume/surface, m
D=4e-6;            % diffusion coefficient, cm^2/s  4e-6, mum^2/ms = cm^2/s*10-8    [Bazhenov 2004]
dx=50*1e-4;       % distance between volume centers, cm
% NA
Nao=130;        % extra Na, mM
Nai=20;         % intra Na, mM
kNa=10;         % NA conversion, 1000 cm^3
% single cell and membrane parametersNe=8100; Ni=1936;  
Cm=0.75;         % mu F/cm^2
e0=26.6393;      % kT/F, in Nernst equation
kappa=10000;     % conductance between compartments, Ohm
S_Soma=0.000001; % cm^2
S_Dend=0.000165; % cm^2
% constants
F=96489;        % coul/M
% somatic
G_Na=3450.0;     % Na, mS/cm^2
G_Kv=200.0;      % Kv-channel mS/cm^2
gg_kl=0.042;     % K leak, mS/cm^2
gg_Nal=0.0198;   % Na leak, mS/cm^2
Vbolz_E=22;        % mV
% dendritic
G_NaD=1.1;       % mS/cm^2
G_NapD=3.5;      % mS/cm^2
G_HVA=0.0195;    % mS/cm^2
G_kl=0.044;      % K leak, mS/cm^2
G_lD=0.01;     % Clleak, mS/cm^2
G_Nal=0.02;      % Na leak, mS/cm^2
E_Ca=140;        % mV
TauCa=800;       % ms
DCa=0.85;        % ???
G_KCa=2.5;       % mS/cm^2
G_Km_E=0.01;       % mS/cm^2
% Pump parametes and glial buffer
Koalpha=3.5;    % mM
Naialpha=20;    % mM
Imaxsoma=25;    % muA/cm^2
Imaxdend=25;    % muA/cm^2
Kothsoma=15;    % mM
koff=0.0008;    % 1/ mM / ms  0.0008 in Bazh model            
K1n=1.0;        % 1/ mM
Bmax=500;       % mM
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
gAMPA_max=1;         % mS/cm^2, estimated from Chizhov 3
gNMDA_max=2;          % mS/cm^2
% refractory period, same for E and I pop
t_ref=2;   
%%

%% SN BORDER
a2=404.5;
b2=-0.8712;
c2=27.9;
d2=-0.1488;

% an approximation
K_SN=0:0.1:100;
CL_SN=a2*exp(b2*K_SN) + c2*exp(d2*K_SN);
 

%% loop over trials
%par
%for p=6:1:6
p=6;
%round(Hz_max/dHz)  % loop over stimulation intensities

tic      % start the timer
    
    tt=0;          % time inside seizure attractor    

% set up stimulation frequency
if p-1==0
    ts=0;
else    
ts=round(1000/((p-1)*dHz));
end

dts=ts;                       % next stimulation interval, ts+dts

HZ(p)=(p-1)*dHz;              % frequency of stimulation, Hz

%% Network ICs
%Npath=[];                                    % KCC2(+)
Npath=450;                                 % KCC2(-)
% KCC2 NORM
  Nnorm=1:1:Ne;
  Nnorm(intersect(Npath,Nnorm))=[];  
  Ikcc2=2*ones(Ne,1);      % KCC2 max current 0.5 norm, 0 path
  Ikcc2(Npath)=0;            % PATHOLOGY
 %Diffusion matrix
 A_diff=M_diff(sqrt(Ne));
 % Indexes of all connected elements
l=zeros(Ne,4);
for t=1:1:Ne 
l(t,1:end)=find(A_diff(t,1:end)>0);
end
 % ICs KCC2(+)
 Bs=random('normal',499.93,0,Ne,1);                    % mM buffer sould not be random
 Cli=subplus(random('normal',3.46,0,Ne,1));            % mM 5.13, Ikcc2=0.5
 Ko=subplus(random('normal',3.35,0,Ne,1));             % mM 3.25
 cai=subplus(random('normal',0,0,Ne,1));               % mM 
 VD=random('normal',-70.04,0,Ne,1);          % mV 
 VSOMA=random('normal',-70.09,0,Ne,1);       % mV 
 m_iKv=subplus(random('normal',0.00,0,Ne,1));        % 1
 m_iNa=subplus(random('normal',0.01,0,Ne,1));        % 1
 h_iNa=subplus(random('normal',0.91,0,Ne,1));        % 1 
 m_iNaD=subplus(random('normal',0.00,0,Ne,1));       % 1
 h_iNaD=subplus(random('normal',0.91,0,Ne,1));       % 1
 m_iNapD=subplus(random('normal',0.00,0,Ne,1));      % 1 
 m_iKCa=subplus(random('normal',0.00,0,Ne,1));       % 1  
 m_iHVA=subplus(random('normal',0.00,0,Ne,1));       % 1
 h_iHVA=subplus(random('normal',0.64,0,Ne,1));       % 1
 m_iKm_E=subplus(random('normal',0.01,0,Ne,1));        % 1 
% ICs KCC2(-)
 Bs(Npath)=random('normal',499.92,0,length(Npath),1);                  % mM buffer sould not be random
 Cli(Npath)=subplus(random('normal',11.30,0,length(Npath),1));         % mM
 Ko(Npath)=subplus(random('normal',3.46,0,length(Npath),1));           % mM
 cai(Npath)=subplus(random('normal',0,0,length(Npath),1));            % mM 
 VD(Npath)=random('normal',-65.37,0,length(Npath),1);                  % mV 
 VSOMA=random('normal',-65.45,0,Ne,1);       % mV 
 m_iKv(Npath)=random('normal',0.00,0,length(Npath),1);                 % 1 
 m_iNa(Npath)=random('normal',0.01,0,length(Npath),1);                 % 1
 h_iNa(Npath)=random('normal',0.83,0,length(Npath),1);                 % 1 
 m_iNaD(Npath)=subplus(random('normal',0.01,0,length(Npath),1));       % 1
 h_iNaD(Npath)=subplus(random('normal',0.84,0,length(Npath),1));       % 1 
 m_iNapD(Npath)=subplus(random('normal',0.00,0,length(Npath),1));      % 1 
 m_iKCa(Npath)=subplus(random('normal',0.00,0,length(Npath),1));       % 1 
 m_iHVA(Npath)=subplus(random('normal',0.00,0,length(Npath),1));       % 1
 h_iHVA(Npath)=subplus(random('normal',0.58,0,length(Npath),1));       % 1
 m_iKm_E=subplus(random('normal',0.01,0,Ne,1));        % 1
 VSOMA_sp=10*ones(Ne,1);   % vector of refractory times, >t_ref
 % REPRESENTATIVE CELLS
 % KCC2 norm
 VEnorm=zeros(1);
 VEnorm_IE=zeros(1);
 VEnorm_II=zeros(1);
 Konorm=zeros(1);
 Clinorm=zeros(1);
 cai_VEnorm=zeros(1); 
 % KCC2 path
  VEpath=zeros(1);
  VEpath_IE=zeros(1);
  VEpath_II=zeros(1); 
  Kopath=zeros(1);
  Clipath=zeros(1);
  cai_VEpath=zeros(1);  
  % EXC Firings
firings_E=[];                                 % spike timings
fired_E=[];                                   % indexes of spikes fired at t
% SYNAPTIC CONNECTIONS
IE=0;
II=0; 
% EE, AMPA
gEE_AMPA=zeros(Ne,1);
% EE, NMDA
gEE_NMDA=zeros(Ne,1);
x_NMDA=zeros(Ne,1);
% EI, AMPA
gEI_AMPA=zeros(Ni,1);
% II, GABA
gII_GABA=zeros(Ni,1);
% IE, GABA
gIE_GABA=zeros(Ne,1);
% GABA
ggGABA_ext=0;
gGABA_ext=0;
% AMPA
ggAMPA_ext=0;
gAMPA_ext=0;
% NMDA
ggNMDA_ext=0;
gNMDA_ext=0;
%%


%% loop over time
for t=1:1:round(T/dt) 
        
%% Network Integration
%Delta-funciton    
    if t*dt<=Tst    % stimulation only during the stimulation period
 if t*dt==ts        % generation of stimulus times
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
    
% ALGEBRAIC EQUATIONS    
 % Na-P-pump
 Ap=(1/((1+(Naialpha/Nai))^3))*(1./((1+(Koalpha./Ko)).^2));
 Ikpump=-2*Imaxsoma*Ap;
 INapump=3*Imaxsoma*Ap; 
 % reversal potentials on soma and dendrite
 % K
 VKe=e0*log(Ko/Ki_e);
 % NA
 VNAe=e0*log(Nao/Nai);
 % CL
 VCL=e0*log(Cli/Clo);
 VGABA=e0*log((4*Cli+HCO3i)./(4*Clo+HCO3o));   
 % GABA ext
 gGABA_ext(t+1)=ggGABA_ext(t)*dt + gGABA_ext(t);
 ggGABA_ext(t+1)=(alpha1_GABA.*alpha2_GABA.*( delta_I.*(1-gGABA_ext(t))./K(alpha1_GABA,alpha2_GABA)-gGABA_ext(t)-(1/alpha1_GABA +1/alpha2_GABA).*ggGABA_ext(t))).*dt +ggGABA_ext(t); 
 % AMPA ext
 gAMPA_ext(t+1)=ggAMPA_ext(t)*dt + gAMPA_ext(t);
 ggAMPA_ext(t+1)=(alpha1_AMPA.*alpha2_AMPA.*( delta_E.*(1-gAMPA_ext(t))./K(alpha1_AMPA,alpha2_AMPA)-gAMPA_ext(t)-(1/alpha1_AMPA +1/alpha2_AMPA).*ggAMPA_ext(t))).*dt +ggAMPA_ext(t);
 % NMDA ext
 gNMDA_ext(t+1)=ggNMDA_ext(t)*dt + gNMDA_ext(t);
 ggNMDA_ext(t+1)=(alpha1_NMDA.*alpha2_NMDA.*( delta_E.*(1-gNMDA_ext(t))./K(alpha1_NMDA,alpha2_NMDA)-gNMDA_ext(t)-(1/alpha1_NMDA +1/alpha2_NMDA).*ggNMDA_ext(t))).*dt +ggNMDA_ext(t);
 % dendrite current
 f_NMDA=1/(1+Mg/3.57*exp(-0.062*VD(450)));
 iDendrite= -G_lD*(VD-VCL) -G_kl*(VD-VKe) -G_Nal*(VD-VNAe) -2.9529*G_NaD*m_iNaD.^3.*h_iNaD.*(VD-VNAe) -G_NapD*m_iNapD.*(VD-VNAe) -G_KCa*m_iKCa.^2.*(VD-VKe) -2.9529*G_HVA*m_iHVA.^2.*h_iHVA.*(VD-E_Ca) -2.9529*G_Km_E*m_iKm_E.*(VD-VKe) -INapump -Ikpump;
 iDendrite(450)=iDendrite(450) -gNMDA_max*gNMDA_ext(t)*f_NMDA*(VD(450)-VNMDA) -gGABA_max*gGABA_ext(t)*(VD(450)-VGABA(450)) -gAMPA_max.*gAMPA_ext(t)*(VD(450)-V_AMPA);
 % somatic voltage
 g1_SOMA=gg_kl +gg_Nal +(2.9529*G_Na*m_iNa.^3.*h_iNa) +(2.9529*G_Kv.*m_iKv);
 g2_SOMA=gg_kl*VKe +gg_Nal*VNAe +(2.9529*G_Na*m_iNa.^3.*h_iNa*VNAe) +(2.9529*G_Kv*m_iKv.*VKe) -INapump -Ikpump;
 VSOMA=(VD + (kappa*S_Soma *g2_SOMA)) ./ (1+kappa*S_Soma*g1_SOMA);
 % IKv Soma
 a_iKv=0.02*(VSOMA-Vbolz_E)./(1-exp(-(VSOMA-Vbolz_E)/9));
 b_iKv=-0.002*(VSOMA-Vbolz_E)./(1-exp((VSOMA-Vbolz_E)/9));
 tauKvm=1./((a_iKv+b_iKv)*2.9529);
 infKvm=a_iKv./(a_iKv+b_iKv);
 % INa Soma
 am_iNa=0.182*(VSOMA-10+35)./(1-exp(-(VSOMA-10+35)/9));
 bm_iNa=0.124*(-VSOMA+10-35)./(1-exp(-(-VSOMA+10-35)/9));
 ah_iNa=0.024*(VSOMA-10+50)./(1-exp(-(VSOMA-10+50)/5));
 bh_iNa=0.0091*(-VSOMA+10-75)./(1-exp(-(-VSOMA+10-75)/5));
 tau_m=(1./(am_iNa+bm_iNa))/2.9529;
 tau_h=(1./(ah_iNa+bh_iNa))/2.9529;
 m_inf_new=am_iNa./(am_iNa+bm_iNa);
 h_inf_new=1./(1+exp((VSOMA-10+65)/6.2));
 % NaP, D current
 minfiNapD = 0.02./(1 + exp(-(VD+42)/5));
 % INa D, sodium channel
 am_iNaD=0.182*(VD-10+35)./(1-exp(-(VD-10+35)/9));
 bm_iNaD=0.124*(-VD+10-35)./(1-exp(-(-VD+10-35)/9));
 ah_iNaD=0.024*(VD-10+50)./(1-exp(-(VD-10+50)/5));
 bh_iNaD=0.0091*(-VD+10-75)./(1-exp(-(-VD+10-75)/5));
 minf_newD = am_iNaD./(am_iNaD+bm_iNaD);
 hinf_newD = 1./(1+exp((VD-10+65)/6.2));
 tau_mD = (1./(am_iNaD+bm_iNaD))/2.9529;
 tau_hD = (1./(ah_iNaD+bh_iNaD))/2.9529;
%%%% iKCa %%%%
 minf_iKCa = (48*cai.^2/0.03)./(48*cai.^2/0.03 + 1);
 taum_iKCa = (1./(0.03*(48*cai.^2/0.03 + 1)))/4.6555;
%%%% IHVA %%%%
 am_iHVA = 0.055*(-27 - VD)./(exp((-27-VD)/3.8) - 1);
 bm_iHVA = 0.94*exp((-75-VD)/17);
 ah_iHVA = 0.000457*exp((-13-VD)/50);
 bh_iHVA = 0.0065./(exp((-VD-15)/28) + 1);
 tauHVAh = 1./((ah_iHVA+bh_iHVA)*2.9529);
 infHVAh = ah_iHVA./(ah_iHVA+bh_iHVA);
 tauHVAm = 1./((am_iHVA+bm_iHVA)*2.9529);
 infHVAm = am_iHVA./(am_iHVA+bm_iHVA);
 %%% IKM %%%%
 am_iKm_E = 0.001*(VD + 30)./(1 - exp(-(VD + 30)/9));
 bm_iKm_E = -0.001*(VD + 30)./(1 - exp((VD + 30)/9));
 tauKmm_E = 1./((am_iKm_E+bm_iKm_E)*2.9529);
 infKmm_E = am_iKm_E./(am_iKm_E+bm_iKm_E);
% ION CURRENTS
 ICL = G_lD*(VD-VCL);
 ICL((450))=ICL((450)) + gGABA_max*gGABA_ext(t)*(VD(450)-VGABA(450));
 IK = gg_kl*(VSOMA-VKe) +G_kl*(VD-VKe) +G_KCa*m_iKCa.*m_iKCa.*(VD-VKe) +2.9529*G_Km_E.*m_iKm_E.*(VD-VKe) +(2.9529*G_Kv*m_iKv.*(VSOMA-VKe))/200;
% GLIA
 kon=koff./(1+exp((Ko-Kothsoma)/(-1.15)));
 Glia=koff*(Bmax-Bs)/K1n -kon/K1n.*Bs.*Ko;
% EULER METHOD
 VD=((1/Cm).*(iDendrite +(VSOMA-VD) / (kappa*S_Dend)))*dt + VD;
 m_iNa=(-(m_iNa-m_inf_new)./tau_m)*dt + m_iNa;
 h_iNa=(-(h_iNa-h_inf_new)./tau_h)*dt + h_iNa;
 m_iKv=(-(m_iKv-infKvm)./tauKvm)*dt + m_iKv;
 m_iNaD =(-(m_iNaD - minf_newD)./tau_mD)*dt +m_iNaD;
 h_iNaD =(-(h_iNaD - hinf_newD)./tau_hD)*dt +h_iNaD;
 m_iNapD=(-(m_iNapD - minfiNapD)./0.1992)*dt +m_iNapD; 
 m_iKCa =(-(1./taum_iKCa).*(m_iKCa - minf_iKCa))*dt + m_iKCa;
 m_iHVA = (-(m_iHVA-infHVAm)./tauHVAm)*dt + m_iHVA;
 h_iHVA = (-(h_iHVA-infHVAh)./tauHVAh)*dt + h_iHVA;
 m_iKm_E = (-(m_iKm_E-infKmm_E)./tauKmm_E)*dt + m_iKm_E; 
% ION CONCENTRATION CHANGES %
 Ko=(kK/F/d*(IK +Ikpump +Ikpump -Ikcc2.*(VKe-VCL)./((VKe-VCL)+Vhalf)) +Glia +D/dx^2.*(sum(Ko(l),2)-4*Ko) )*dt + Ko;
 Cli=kCL/F*(ICL +Ikcc2.*(VKe-VCL)./((VKe-VCL)+Vhalf) )*dt + Cli;
 cai=(-5.1819e-5*2.9529*G_HVA*m_iHVA.^2.*h_iHVA.*(VD-E_Ca)/DCa + (0.00024-cai)./TauCa)*dt + cai; 
 % Glial buffer
 Bs=(koff.*(Bmax-Bs) -kon.*Bs.*Ko)*dt + Bs;   
 fired_E=find(VSOMA>=0);                   % indices of spikes in the network for one time step  
 % Processing of fired_E
 INT_E=intersect(find(VSOMA_sp<t_ref),fired_E);   % intersection VSOMA_sp<2 and fired_E
 fired_E=setxor(fired_E,INT_E);                   % remove fired_E elements that intersect with VSOMA_sp<2
 VSOMA_sp=VSOMA_sp + dt;                          % update time for t* vector
 VSOMA_sp(fired_E)=0; 
% Record firings
 firings_E=[firings_E; t+0*fired_E,fired_E]; 
 % REPRESENTATIVE CELL
 VEnorm(t)=VSOMA(450);
 Konorm(t)=Ko(450);
 Clinorm(t)=Cli(450);
 VEnorm_IE(t)=gEE_AMPA(450)*(V_AMPA-VD(450));
 VEnorm_II(t)=gIE_GABA(450)*(VGABA(450)-VD(450));
 cai_VEnorm(t)=cai(450); 
%%

%% SN border approximation
     Cl_appr=a2*exp(b2*Konorm(t)) + c2*exp(d2*Konorm(t)); % test neuron          
     if Clinorm(t)>Cl_appr                          % test neuron in the bursting regime or not
        tt=tt+1;     
     end   
 
end       % loop over time

time=(1:1:round(T/dt))*dt;
SIMULATION_TIME=toc % end of timer
% seizure time
%Tseizure(p)=tt*dt;

%{
%%
figure('units','normalized','outerposition',[0 0 1 1]);

subplot(2,2,1);
imagesc((reshape(Ko,sqrt(Ne),sqrt(Ne)))'); % ,[3 8]
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

subplot(2,2,2);
imagesc((reshape(Cli,sqrt(Ne),sqrt(Ne)))'); % ,[5 25]
set(gca,'Ydir','normal');
ylabel('Cell index');
xlabel('Cell index');
%imagesc((reshape(VGABA,sqrt(Ne),sqrt(Ne)))');
set(gca,'FontSize',10);             % set the axis with big font
title('Cl_{IN}, mM');
%title('VGABA, mV');
colorbar;
box off;

subplot(2,2,3);
plot(Konorm,Clinorm,'Linewidth',2); hold on
plot(K_SN,CL_SN,'Linewidth',2,'Color','black');
axis([0 30 0 30])
hold on
axis([0 30 0 30]);
set(gca,'FontSize',10);             % set the axis with big font
xlabel('Ko, mM');
ylabel('Cli, mM');
title('KCC2(+)');
box off;

subplot(2,2,4);
plot(time,VEnorm,time,VEpath);
set(gca,'FontSize',10);             % set the axis with big font
ylabel('V_E, mV');
box off;
%%

saveas(gcf,sprintf('Bazh_Ko_Cli_diff_net_PATH_%d.jpg',HZ(p)),'jpg');  % save *.jpg
%}
%parsave(sprintf('Bazh_Ko_Cli_diff_net_NORM_%d.mat',HZ(p)),Tseizure(p),HZ(p),Npath,Nnorm,time,SIMULATION_TIME,dt,Ne,Ni,firings_E,Konorm,Kopath,Ko,Clinorm,Clipath,Cli,VEnorm,VEpath);           % save *.mat
parsave(sprintf('Bazh_Ko_Cli_diff_net_NORM_%d.mat',HZ),Tseizure,HZ,Npath,Nnorm,time,SIMULATION_TIME,dt,Ne,Ni,firings_E,Konorm,Kopath,Ko,Clinorm,Clipath,Cli,VEnorm,VEpath);           % save *.mat
%close;

%end       % loop over frequencies

%{
figure;

%% VGABA-intensity plot
plot(HZ,Tseizure,'blue','LineWidth',6);
axis([0 max(HZ) 0 T])
set(gca,'FontSize',30);
xlabel('Intensity of stimulation, Hz');
ylabel('Afterdischarge duration, ms');
title('KCC2()');

%% Save to file
save('KCC2(+).mat');
%}
