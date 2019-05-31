Tr=20;

for i=1:1:length(aLFP)
    
    for j=1:1:length(aLFP)
     
    [freq,psdx,Hz,p_amp] = spect_peak(aLFP{i,j},dt/1000,50); % get all the spectrum    
    title(sprintf('%d AMPA %d GABA',i,j));
            PSD{i,j}=psdx;
            FR{i,j}=freq;
    
        if  p_amp>=Tr                     % if peak amplitude > Tr
            maxFR(i,j)=Hz;      % frequency location of the peak
            maxPeak(i,j)=p_amp;   % value of the peak                
        end
    
    end

end    