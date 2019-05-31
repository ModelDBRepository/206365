function [ peak_t,peak_v,peak_f ] = spect_scan(LFP,T,Tr,dt)

%   Analyses the spectrum peak with non-overlap windows ot T size
%   if it larger than threshould Tr
%   indicates the value of the timing, value and frequency of the peak
%   as peak_t, peak_v, peak_f

% Tr=35 for seizure indentefication

t_old=1;
dT=T;             % shift every second dT
a=0;

peak_t=zeros;        % timings of the peaks
peak_v=zeros;        % peak value
peak_f=zeros;        % peak frequency location


for t=1:length(LFP)
   
    if t*dt==T
    
    a=a+1;         % counter for the sweeps
    
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
        
    T=T+dT;         % time for the next sweep 
    t_old=t;        % moment of the previous sweep
    
    end
    
end


end

