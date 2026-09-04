function data_rx = vlc_channel(data_in, SNR, Fs, factor, ifNL)
data_tx = data_in;
%% Nonlinear
if ifNL == 1
    Vpp = 1.2;
    x = data_tx/(max(data_tx)-min(data_tx))*2*Vpp;
    % data_tx=3.036./(1+exp(-1.946.*x))-1.563; % strong NL
    data_tx = 4.412./(1+exp(-1.07.*x))-2.206;  % weak NL
    data_tx = data_tx./sqrt(mean(abs(data_tx).^2));
    figure
    plot(data_in, data_tx)
end
%% Frequency fading

n = 1:length(data_tx)/2;
fs = 2*Fs/(length(data_tx));
fsn = fs*n;
ch1 = exp(-fsn/factor);
ch2 =  fliplr(ch1);
ch = [ch1 ch2];
% chall=repmat(ch,1,[]);

data_ch_fft = fft(data_tx);
data_ch_after = data_ch_fft.*ch;
data_ch_ifft = real(ifft(data_ch_after));
data_tx = data_ch_ifft - mean(data_ch_ifft);

data_rx = awgn(data_tx, SNR, 'measured');

