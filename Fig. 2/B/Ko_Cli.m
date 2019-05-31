
plot(Ko1,Cli1,Ko2,Cli2);
hold on
plot(K_eq(1),Cl_eq(1),'.','MarkerSize',40);
hold on
plot(K_eq(2),Cl_eq(2),'.','MarkerSize',40);
box off

set(gca,'FontSize',40);             % set the axis with big font
xlabel('Ko (mM)');
ylabel('Cli (mM)');