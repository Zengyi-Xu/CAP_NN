function data_af_db = DB_tran_code(data_be_db, Num)

M = Num;
n = length(data_be_db);
data_ori = data_be_db;
data_ori_I = (real(data_ori)+(M-1))/2;
data_ori_Q = (imag(data_ori)+(M-1))/2;
data_diff_I = zeros(n, 1);

for k = 2:n
    data_diff_I(k) = data_ori_I(k) - data_diff_I(k-1);
end
data_mod_I = mod(data_diff_I, M);
data_duo_I = zeros(n, 1);
% data_mod_I=real(data_ori);
for k = 2:n
    data_duo_I(k) = data_mod_I(k) + data_mod_I(k-1);
end

data_diff_Q = zeros(n, 1);
for k = 2:n
    data_diff_Q(k)=data_ori_Q(k)-data_diff_Q(k-1);
end
data_mod_Q = mod(data_diff_Q, M);
data_duo_Q = data_mod_Q;
% data_mod_Q=imag(data_ori);
for k = 2:n
    data_duo_Q(k) = data_mod_Q(k)+data_mod_Q(k-1);
end

data_duo_I = data_duo_I-(M-1);
data_duo_Q = data_duo_Q-(M-1);

% eyediagram(data_duo_I,2)
% eyediagram(data_duo_Q,2)
% figure()
% plot(data_duo_I,data_duo_Q,'*')

data_duo = complex(data_duo_I, data_duo_Q);

% figure;
% plot(data_ori_I,'b*');
% hold on;
% plot(data_rx_I,'r.');

data_af_db = data_duo;
end