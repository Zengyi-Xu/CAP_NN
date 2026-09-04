function [complex_data, rxdatasym, ber]=CAPmatch_filter(datain, gt, t, f, taps, upsampleno, offsetsample, varargin)
gtI = gt.*cos(2*pi*f*t);
gtQ = gt.*sin(2*pi*f*t);
% gtI = gtI((1:taps)+length(t)/2-(taps-1)/2);
% gtQ = gtQ((1:taps)+length(t)/2-(taps-1)/2);
datain = reshape(datain, 1, []);
data = [datain((end-(taps-1)/2+1):end), datain, datain(1:(taps-1)/2)];
I = conv(data, gtI);
Q = conv(data, gtQ);
dataCapI = I(taps:(end-taps+1));
dataCapQ = Q(taps:(end-taps+1));
DataCap = dataCapI + 1i*dataCapQ;
receiveddataXY_sa = downsample(DataCap, upsampleno, offsetsample);
receiveddataXY_sa = receiveddataXY_sa - mean(receiveddataXY_sa);
receiveddataXY_sa = receiveddataXY_sa/sqrt(mean(abs(receiveddataXY_sa).^2));
complex_data = receiveddataXY_sa;
if nargin == 9
    recoverdata = reshape(complex_data,1,[]);
    QAMorder = varargin{1};
    x = 0:QAMorder-1;
    x = qammod(x,QAMorder);
    Avt = sqrt(mean(abs(x).^2));
    recoverdata = recoverdata.*Avt;
    % plot_hist3nan(recoverdata1);
    rxdatasym = qamdemod(recoverdata, QAMorder);% ¼ÆËãBER
    origindata = varargin{2};
    [~,ber] = biterr(rxdatasym', origindata);
    % disp(['Ber without LMS is ', num2str(ber)]);
end
