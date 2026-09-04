% modified to be compatible for DB
function [complex_data, rxdatasym, ber] = mdb_match_filter(datain, gt1, t, f, upsampleno, offsetsample, varargin)
%% down convertion
down_data = datain.*cos(2*pi*f*t) - 1i*datain.*sin(2*pi*f*t);
%% filter
band_data = conv(down_data, gt1, 'same');
%% down sample
band_data = downsample(band_data, upsampleno, offsetsample);
band_data = band_data - mean(band_data);
band_data = band_data/sqrt(mean(abs(band_data).^2));
complex_data = band_data;
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
