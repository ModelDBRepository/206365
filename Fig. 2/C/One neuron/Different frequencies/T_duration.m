plot(HZ,Tseizure_norm,'blue','Linewidth',2); hold on
plot(HZ,Tseizure_path,'red','Linewidth',2); hold on
plot(HZ,Tseizure_norm_diff,'--b','Linewidth',2); hold on
plot(HZ,Tseizure_path_diff,'--r','Linewidth',2); hold on


axis([0 max(HZ) 0 T])
set(gca,'FontSize',30);
xlabel('Intensity of stimulation, Hz');
ylabel('Afterdischarge duration, s');
box off;
legend('KCC2(+)','KCC2(-)','KCC2(+) Network diffusion','KCC2(-) Network diffusion');
