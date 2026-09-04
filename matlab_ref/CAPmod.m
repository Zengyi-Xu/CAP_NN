% CAP modulation
function [dataout] = CAPmod(complex_sym, gt1, t, f1, taps, upsampleno)
gtI1 = gt1.*cos(2*pi*f1*t);
gtQ1 = gt1.*sin(2*pi*f1*t);
% gtI1 = gtI1((1:taps)+length(t)/2-(taps-1)/2);
% gtQ1 = gtQ1((1:taps)+length(t)/2-(taps-1)/2);

qamdata1 = reshape(complex_sym, 1, length(complex_sym));
% upsampleno = 4;
% QAMorder = 16;

% qamdata1 = qammod(complex_sym, QAMorder);

Idata1 = real(qamdata1);
Idata1 = upsample(Idata1, upsampleno);
Qdata1 = imag(qamdata1);
Qdata1 = upsample(Qdata1, upsampleno);

%% pulse shaping 
Idata1 = [Idata1((end-(taps-1)/2+1):end), Idata1, Idata1(1:(taps-1)/2)];
Qdata1 = [Qdata1((end-(taps-1)/2+1):end), Qdata1, Qdata1(1:(taps-1)/2)];

I = conv(Idata1, gtI1);
Q = conv(Qdata1, gtQ1);

DataCapI = I(taps:(end-taps+1));
DataCapQ = Q(taps:(end-taps+1));

%% CAP
DataCap = DataCapI - DataCapQ;
DataCap = DataCap/sqrt(mean(abs(DataCap).^2));
dataout = reshape(DataCap,1,[]);

