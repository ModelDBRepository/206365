subplot(2,1,1)
plot(Konorm,Clinorm,Ko_SN,Cli_SN,'.',Konorm(1),Clinorm(1),'.','Markersize',30);
axis([0 10 0 15]);
title('KCC2(+)');
set(gca,'FontSize',30);             % set the axis with big font
xlabel('Ko, mM');
ylabel('Cli, mM');
box off;

subplot(2,1,2);
plot(time,VEnorm);
set(gca,'FontSize',30);             % set the axis with big font
xlabel('time, ms');
ylabel('V_{S}, mV');
box off;