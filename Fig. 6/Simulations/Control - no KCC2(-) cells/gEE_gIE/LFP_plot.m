

for i=1:1:length(aLFP)    
    for j=1:1:length(aLFP)
        figure('units','normalized','outerposition',[0 0 0.25 0.25]);
        plot((1:1:length(aLFP{i,j}))*dt,aLFP{i,j});
        xlabel('time, ms');
        ylabel('LFP, \muV/cm^2');
        title(sprintf('%d AMPA %d GABA',i,j));
    end
end