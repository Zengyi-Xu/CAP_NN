%  CAP code for DB
function [dataout] = mdb_mod(complex_sym, gt1, t, f1, upsampleno)
qamdata1 = reshape(complex_sym, 1, length(complex_sym));
%% pulse shaping
up_data = upsample(qamdata1, upsampleno);
data = conv(up_data, gt1, 'same');
%% cap data
data_cap = real(data).*cos(2*pi*f1*t) - imag(data).*sin(2*pi*f1*t);

data_cap = data_cap/sqrt(mean(abs(data_cap).^2));
dataout = reshape(data_cap, 1, []);
