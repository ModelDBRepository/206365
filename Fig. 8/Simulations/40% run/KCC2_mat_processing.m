for i=1:1:14

PATH=ceil((i*2)^2/841*100);                                 % pecrcentage KCC2(-) cells
    
load(sprintf('KCC2(-)%d_LFP.mat',PATH));

%sprintf('peak_t_%d',PATH)=peak_t;

end