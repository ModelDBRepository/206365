

%%
K_st=Ko(find(Ko<=SN(1)));
VD_st=VD(find(VD<=SN(2)));


l=length(VD_st);
plot(K_st(1:l),VD_st(1:l),'red');

hold on
plot(Ko(l:end),VD(l:end),'green');

set(gca,'FontSize',40);             % set the axis with big font
xlabel('Ko (mM)');
ylabel('Voltage (mV)');

box off