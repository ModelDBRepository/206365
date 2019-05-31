
%% frequency value of the peak
%peak=[0,0,0,0,0,0,min((peak_f_18(find(peak_f_18>0)))),min((peak_f_24(find(peak_f_24>0)))),min((peak_f_31(find(peak_f_31>0)))),min((peak_f_39(find(peak_f_39>0)))),min((peak_f_48(find(peak_f_48>0)))),min((peak_f_58(find(peak_f_58>0)))),min((peak_f_69(find(peak_f_69>0)))),min((peak_f_81(find(peak_f_81>0)))),min((peak_f_94(find(peak_f_94>0)))),min((peak_f_100(find(peak_f_100>0))))];
KCC2=[ceil(((0:1:14)*2).^2/841*100),100];

% timing of the peak
peak_time=[0,0,0,0,0,0,min((peak_t_18(find(peak_t_18>0)))),min((peak_t_24(find(peak_t_24>0)))),min((peak_t_31(find(peak_t_31>0)))),min((peak_t_39(find(peak_t_39>0)))),min((peak_t_48(find(peak_t_48>0)))),min((peak_t_58(find(peak_t_58>0)))),min((peak_t_69(find(peak_t_69>0)))),min((peak_t_81(find(peak_t_81>0)))),min((peak_t_94(find(peak_t_94>0)))),min((peak_t_100(find(peak_t_100>0))))];



%% Time to seizure plot
figure;
plot(KCC2,peak_time,'blue');
set(gca,'FontSize',30);             % set the axis with big font
ylabel('Time, ms');
xlabel('%,  KCC2(-)');
title('Time to seizure');
box off;


%% Seizure frequency plot
figure;
plot(KCC2,peak,'blue');
set(gca,'FontSize',30);             % set the axis with big font
ylabel('Frequency, Hz');
xlabel('%,  KCC2(-)');
title('First peak frequency');
box off;
