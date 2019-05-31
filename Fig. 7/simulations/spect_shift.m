
for i=1:1:14

PATH=ceil((i*2)^2/841*100);                                 % pecrcentage KCC2(-) cells

load(sprintf('Net_burst_extKo_IN_path_%d.mat',PATH));    
    
T=5000;           % first duration
t_old=1;
dT=T;           % shift every second dT
a=0;

peak_T=35;          % threshould for the peak value on the spectrum

peak_t=zeros;        % timings of the peaks
peak_v=zeros;        % peak value
peak_f=zeros;        % peak frequency location

for t=1:length(time)
   
    if t*dt==T
    
    a=a+1;         % counter for the sweeps
    % figure(a);
    [A,B,C,D]=spect_peak(LFP(t_old:t),dt/1000,50);
            
    if D>peak_T
    peak_t(a)=t*dt;
    peak_v(a)=D;        % peak value
    peak_f(a)=C;        % peak frequency location
    else
        peak_t(a)=0;
        peak_v(a)=0;
        peak_f(a)=0;
    end
    
    if  t_old==1
        t_old=0;
    end
    
    %title(sprintf('%d - %d time s',t_old*dt/1000, t*dt/1000));
    %ylim([0 6]);
    
    T=T+dT;         % time for the next sweep 
    t_old=t;        % moment of the previous sweep
    
    end
    
end

%%
figure('units','normalized','outerposition',[0 0 1 1]); % full screen

path=ceil(length(Npath)/841*100);
%%
subplot(3,1,1);
plot(peak_t,peak_f,'.','Markersize',10);
title(sprintf('KCC2(-) %d ',path));
xlim([0 time(end)]);
set(gca,'Fontsize',10);
xlabel('time, ms');
ylabel('Peak frequency, Hz');

subplot(3,1,2);
plot(peak_t,peak_v,'.','Markersize',10);
xlim([0 time(end)]);
set(gca,'Fontsize',10);
xlabel('time, ms');
ylabel('Peak amlitude');

subplot(3,1,3);
plot(time,LFP);
set(gca,'Fontsize',10);
xlabel('time, ms');
ylabel('LFP, \muV');
%%
%
savefig(sprintf('KCC2(-)%d_LFP.fig',path));           % safe *.fig

saveas(gcf,sprintf('KCC2(-)%d_LFP.jpg',path),'jpg');  % save *.jpg

save(sprintf('KCC2(-)%d_LFP.mat',path),'peak_f','peak_t','peak_T','peak_v','path','T','time','LFP');

close
end

%}