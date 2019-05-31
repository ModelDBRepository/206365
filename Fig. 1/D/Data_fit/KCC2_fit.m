plot(x_PATH,y_PATH,'.',x_PATH_fit,y_PATH_fit,x_NORM,y_NORM,'*',x_NORM_fit,y_NORM_fit);

set(gca,'FontSize',30);             % set the axis with big font
xlabel('Membrane potential, mV');
ylabel('PSP, mV');
box off;
