figure('units','normalized','position',[.1 .1 .8 .8]);

% PLOT
subplot(2,2,1);
imagesc((reshape(VSOMA,sqrt(Ne),sqrt(Ne)))');
% , [-80 60]
% , [-67.5 -63.5]
axis image;
set(gca,'FontSize',20);             % set the axis with big font
set(gca,'Ydir','normal');
title(sprintf('E population, T=%dms',Tframe-dTframe));
xlabel('cell index');
ylabel('cell index');
h=colorbar;
h.Label.String = 'Voltage, mV';
colormap jet;

subplot(2,2,2);
imagesc((reshape(VI,sqrt(Ni),sqrt(Ni)))');
%, [-80 60]
% , [-70 -68.2]
axis image;
set(gca,'FontSize',20);             % set the axis with big font
set(gca,'Ydir','normal');
title(sprintf('I population, T=%dms',Tframe-dTframe));
set(gca,'FontSize',20);             % set the axis with big font
xlabel('cell index');
ylabel('cell index');
h1=colorbar;
h1.Label.String = 'Voltage, mV';


subplot(2,2,3);
imagesc((reshape(Ko,sqrt(Ne),sqrt(Ne)))');
% ,[3 5]
% , [3.16 3.17]
axis image;
%pcolor((reshape(Ko,sqrt(Ne),sqrt(Ne)))');
set(gca,'FontSize',20);             % set the axis with big font
set(gca,'Ydir','normal');
title(sprintf('K^{+}_{Out}, T=%d ms',Tframe-dTframe));
xlabel('cell index');
ylabel('cell index');
h2=colorbar;
h2.Label.String = 'Concentration, mM';

subplot(2,2,4);
imagesc((reshape(Cli,sqrt(Ne),sqrt(Ne)))');
% ,[5 20]
% , [3 12]
axis image;
%pcolor((reshape(Cli,sqrt(Ne),sqrt(Ne)))');
set(gca,'FontSize',20);             % set the axis with big font
set(gca,'Ydir','normal');
title(sprintf('Cl^{-}_{In}, T=%dms',Tframe-dTframe));
xlabel('cell index');
ylabel('cell index');
h3=colorbar;
h3.Label.String = 'Concentration, mM';