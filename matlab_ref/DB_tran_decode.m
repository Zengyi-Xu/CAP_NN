function Data_af_DeDB = DB_tran_decode(data_af_db, M)

data_rx = data_af_db;
data_rx_I = round(real(data_rx)+(M-1));
data_rx_Q = round(imag(data_rx)+(M-1));
% data_rx_I=real(data_rx)+(M-1);
% data_rx_Q=imag(data_rx)+(M-1);
data_rx_I = mod(data_rx_I, M);
data_rx_Q = mod(data_rx_Q, M);
data_rx_I = data_rx_I*2-(M-1);
data_rx_Q = data_rx_Q*2-(M-1);
data_rx = complex(data_rx_I, data_rx_Q);
% figure;
% plot(data_ori_I,'b*');
% hold on;
% plot(data_rx_I,'r.');

Data_af_DeDB = data_rx;
end