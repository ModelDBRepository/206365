function [ freq,psdx,Hz,f_max] = spect_peak(x,dt,max_f)

% computes the power spectrum (normalized by the peak) of the signal x
% sampling rate is calculated by dt in s
% max_f determines the maximal frequency in Hz to show

% output Hz - frequency location of the max peak
%        f_max - value of the peak

%%
N = length(x);        % length of the signal
Fs=1/dt;         % sampling frequency

xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(x):Fs/2;

max(find(freq<=0.5)); % index of the frequency that correspond to 0.5 Hz

shift=max(find(freq<=0.5));         % 100 to get rid of the nonsense peak for low freq

f_max=max(psdx(shift:end));         % maximum peak value on the spectrum

f_max_hz=find(psdx==max(psdx(shift:end))); % maximal peak freq location

Hz=freq(f_max_hz);     % peak frequency

% plot the spectrum
%{
figure;
plot(freq,psdx);                    % /f_max, normalization
axis([1 max_f 0 f_max]);

grid on
title('Spectrum using FFT')
xlabel('Frequency (Hz)')
ylabel('Power')
%}


%%
end

