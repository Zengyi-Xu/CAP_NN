clc 
clear all
close all
%% multi-band system parameter
% ²Î¿¼Non-Orthogonal Multi-band CAP for Highly Spectrally Efficient VLC Systems
% ÖÐµÄ²ÎÊýÉèÖÃ

numofsymbols = 1024*32; % ·¢ËÍµÄ·ûºÅÊý
% Bcap = 300e6; % ¶à´ø×ÜµÄ´ø¿í
Rs = 300e6;
rolloff = 0.2; % ¹ö½µÏµÊý
Bcap = Rs*(1 + rolloff);
% Rs = Bcap/(1+rolloff); % ¶à´ø×ÜµÄ²¨ÌØÂÊ
cf = 0.11; % Ñ¹Ëõ±ÈÀý
M = 16; 
m = 3; %×Ó´øÊý
eta = log(M)/log(2)/((1+rolloff)*(1-cf)); % ÆµÆ×Ð§ÂÊ£¬À´×ÔÎÄÕÂ¹«Ê½
Rs_perband = Rs/m; % Ã¿¸ö×Ó´øµÄ²¨ÌØÂÊ
% upsampleno = ceil(2*m*(1+rolloff))+1; % ÉÏ²ÉÑù±¶Êý
% fs_show = Rs*upsampleno/m; % ²ÉÑùÂÊ
fs = 1.2e9;  %Ç¿ÖÆ¸ÄÎª´ËÖµ£¬·½±ãÊµÑé¿ªÕ¹
upsampleno = round(m*fs/Rs);
% upsampleno = m*ceil(fs/Rs);
% upsampleno = m*4;
fc = zeros(1, m); % Ã¿¸ö×Ó´øµÄÖÐÐÄÆµÂÊ
% f_start = 20e6*fs/fs_show; % ±Ü¿ªµÍÆµ£¬ÆµÂÊ¿ªÊ¼µã
for n = 1:m
    % Paul Haigh<Visible light communications: multi-band super-Nyquist CAP modulation>ÖÐµÄ¹«Ê½
    fc(n) = Bcap/(2*m)-(n-1)*(Bcap/m-Bcap*(1-cf))/(m-1); 
end
% fc = fc + f_start;
%% multi-band
% S = RandStream.getGlobalStream;
% QAMorder = [M, M, M];
% taps = 8 * upsampleno + 1;
% rng_num = [12, 22, 32];
% [data_mapping(1,:),data1, cap_signal_1, filt1_I_path, filt1_Q_path,var_1] = cap_gen(QAMorder(1), Rs_perband, numofsymbols, ...
%     upsampleno,taps,rolloff, fc(1), rng_num(1));
% [data_mapping(2,:),data2, cap_signal_2, filt2_I_path, filt2_Q_path,var_2] = cap_gen(QAMorder(2), Rs_perband, numofsymbols, ...
%     upsampleno,taps,rolloff, fc(2), rng_num(2));
% [data_mapping(3,:),data3, cap_signal_3, filt3_I_path, filt3_Q_path,var_3] = cap_gen(QAMorder(3), Rs_perband, numofsymbols, ...
%     upsampleno,taps,rolloff, fc(3), rng_num(3));
% cap_signal_123 = cap_signal_1 + cap_signal_2 + cap_signal_3;
% cap_signal_123 = cap_signal_123./sqrt(mean(abs(cap_signal_123).^2));%¹éÒ»»¯´æ´¢
span = 8;
taps = span * upsampleno + 1;
delay = span*upsampleno/2;
t = (-delay:delay) / fs;
QAMorder = M;
% SRRC
gt1 = shaping_fildes(rolloff, span, upsampleno, 'srrc');
rng(1)
decimal_data1 = randi(QAMorder, numofsymbols, 1)-1;
complex_data1 = qammod(decimal_data1, M);
rng(2)
decimal_data2 = randi(QAMorder, numofsymbols, 1)-1;
complex_data2 = qammod(decimal_data2, M);
rng(3)
decimal_data3 = randi(QAMorder, numofsymbols, 1)-1;
complex_data3 = qammod(decimal_data3, M);
cap_signal_1 = CAPmod(complex_data1, gt1, t, fc(1), taps, upsampleno);
cap_signal_2 = CAPmod(complex_data2, gt1, t, fc(2), taps, upsampleno);
cap_signal_3 = CAPmod(complex_data3, gt1, t, fc(3), taps, upsampleno);
cap_signal_123 = cap_signal_1 + cap_signal_2 + cap_signal_3;
cap_signal_123 = cap_signal_123./sqrt(mean(abs(cap_signal_123).^2));
figure
plot(cap_signal_1(1:72), 'r-', 'linewidth', 2);hold on
plot(cap_signal_2(1:72), 'b-', 'linewidth', 2);hold on
plot(cap_signal_3(1:72), 'k-', 'linewidth', 2);
%% ÐÅµÀ£¬¼ÓÔëÉù
SNR  = 25;
factor = 15;
% cap_signal_123 = vlc_channel(cap_signal_123, SNR, 100, factor, 1);
save('up123_data_for_dnn.txt', 'cap_signal_123', '-ascii');
% cap_signal_123 = awgn(cap_signal_123, SNR, 'measured');
% »­Í¼¹Û²ìÆµÆ×
% awg_SampleRate = fs;
% xfre = ((0:1:length(cap_signal_1)-1)-length(cap_signal_1)/2)*awg_SampleRate/1e6/length(cap_signal_1);
% DataCap1_f = fftshift(fft(cap_signal_1));%µÚ1Â·ÆµÆ×Í¼
% DataCap2_f = fftshift(fft(cap_signal_2));%µÚ2Â·ÆµÆ×Í¼
% DataCap3_f = fftshift(fft(cap_signal_3));%µÚ3Â·ÆµÆ×Í¼
% smth_DataCap1_f = smooth(abs(DataCap1_f),50);%Æ½»¬
% smth_DataCap2_f = smooth(abs(DataCap2_f),50);
% smth_DataCap3_f = smooth(abs(DataCap3_f),50);
% figure;plot(xfre,20*log10(smth_DataCap1_f),'b-');
% hold on;plot(xfre,20*log10(smth_DataCap2_f),'r-');
% hold on;plot(xfre,20*log10(smth_DataCap3_f),'g-');
% ylim([30,70]);%¿ØÖÆYÖá·¶Î§
% title('Frequency Response');xlabel('Bandwidth(MHz)');ylabel('Amplitude(dB)')
figure
plot(20*log10(fftshift(abs(fft(cap_signal_123)))));
%% Rx. Step1.1: extract the corresponding band signals from the multi-band signals.
disp('»ìºÏÐÅºÅ£¬·Ö±ðÊ¹ÓÃf1,f2,f3ÂË²¨Æ÷ÂË²¨');
% [cons_123_filt1] = cap_match_filter(cap_signal_123, filt1_I_path, filt1_Q_path,taps,upsampleno_mband);%¼ÆËãÐÇ×ùµã
% [cons_123_filt2] = cap_match_filter(cap_signal_123, filt2_I_path, filt2_Q_path,taps,upsampleno_mband);
% [cons_123_filt3] = cap_match_filter(cap_signal_123, filt3_I_path, filt3_Q_path,taps,upsampleno_mband);
cons_123_filt1 = CAPmatch_filter(cap_signal_123, gt1, t, fc(1), taps, upsampleno, 0);
cons_123_filt2 = CAPmatch_filter(cap_signal_123, gt1, t, fc(2), taps, upsampleno, 0);
cons_123_filt3 = CAPmatch_filter(cap_signal_123, gt1, t, fc(3), taps, upsampleno, 0);
% ·ù¶È¹éÒ»»¯£¨ÒÑ¾­ºÏ²¢µ½cap_match_filter´úÂëÀï£©
% ºÏ²¢ÎªÒ»¸ö¾ØÕó£¬·½±ãºóÃæICA´úÂëµ÷ÓÃÇó½â»ìºÏ
Tx1 = complex_data1;
Tx2 = complex_data2;
Tx3 = complex_data3;
Rx1 = cons_123_filt1.';
Rx2 = cons_123_filt2.';
Rx3 = cons_123_filt3.';
figure
subplot(221)
plot(Rx1, '.');
subplot(222)
plot(Rx2, '.');
subplot(223)
plot(Rx3, '.');
% x = [real(Rx1), imag(Rx1), real(Rx2), imag(Rx2), real(Rx3), imag(Rx3)];
y = [real(Tx1), imag(Tx1), real(Tx2), imag(Tx2), real(Tx3), imag(Tx3)];
% save('xdata_for_dnn.txt', 'x', '-ascii');
save('ydata_for_dnn.txt', 'y', '-ascii');
ratio = 0.5;
start = round(numofsymbols*ratio)+1;
[ber_raw(1),ser_raw(1)] = MonteCarlo_BER_Estimation(Tx1(start:end), Rx1(start:end), QAMorder);
[ber_raw(2),ser_raw(2)] = MonteCarlo_BER_Estimation(Tx2(start:end), Rx2(start:end), QAMorder);
[ber_raw(3),ser_raw(3)] = MonteCarlo_BER_Estimation(Tx3(start:end), Rx3(start:end), QAMorder);
ber_raw_avg = mean(ber_raw);
disp(['BER raw is: ',num2str(ber_raw_avg)]);
%% NN
addpath('D:\1ÊµÑéÊÒÎÄ¼þ\SCAP_DNN\SCAP_DNNpy2');
predict_data = load('predict_data.txt');
Rx1 = predict_data(:, 1) + 1i*predict_data(:, 2);
Rx2 = predict_data(:, 3) + 1i*predict_data(:, 4);
Rx3 = predict_data(:, 5) + 1i*predict_data(:, 6);
figure
subplot(221)
plot(Rx1, '.');
subplot(222)
plot(Rx2, '.');
subplot(223)
plot(Rx3, '.');
% 
[ber(1),ser(1)] = MonteCarlo_BER_Estimation(Tx1(start:end), Rx1, QAMorder);
[ber(2),ser(2)] = MonteCarlo_BER_Estimation(Tx2(start:end), Rx2, QAMorder);
[ber(3),ser(3)] = MonteCarlo_BER_Estimation(Tx3(start:end), Rx3, QAMorder);
ber_wo_RLS = mean(ber);
disp(['BER with DNN is: ',num2str(ber_wo_RLS)]);
% figure;
% title('½ÓÊÕµÄ¶à´øCAPÐÅºÅÖÐÊ¹ÓÃ²»Í¬ÖÐÐÄÆµÂÊµÄ³ÉÐÎÂË²¨Æ÷½âÐÇ×ùµã');
% subplot 221;plot(real(cons_123_filt1),imag(cons_123_filt1),'b.');
% subplot 222;plot(real(cons_123_filt2),imag(cons_123_filt2),'k.');
% subplot 223;plot(real(cons_123_filt3),imag(cons_123_filt3),'g.');
%% Step2: Ê¹ÓÃ¶ÔÓ¦ÖÐÐÄÆµÂÊ¶ÔÊ±ÓòÐÅºÅ½øÐÐÖØ½¨
cons_123_stage1 = [cons_123_filt1; cons_123_filt2; cons_123_filt3];
% [A1,B1, signal_123_band1,C1,D1, var_rc1] = cap_gen(QAMorder(1), Rs_perband, numofsymbols, upsampleno_mband,taps,rolloff, ...
%     fc(1), rng_num(1), cons_123_filt1);%ÖØ½¨Ê±ÓòÐÅºÅ
% [A2,B2, signal_123_band2,C2,D2, var_rc2] = cap_gen(QAMorder(2), Rs_perband, numofsymbols, upsampleno_mband,taps,rolloff, ...
%     fc(2), rng_num(2), cons_123_filt2);
% [A3,B3, signal_123_band3,C3,D3, var_rc3] = cap_gen(QAMorder(3), Rs_perband, numofsymbols, upsampleno_mband,taps,rolloff, ...
%     fc(3), rng_num(3), cons_123_filt3);
signal_123_band1 = CAPmod(cons_123_filt1, gt1, t, fc(1), taps, upsampleno);
signal_123_band2 = CAPmod(cons_123_filt2, gt1, t, fc(2), taps, upsampleno);
signal_123_band3 = CAPmod(cons_123_filt3, gt1, t, fc(3), taps, upsampleno);
% »­Í¼
% figure;
% subplot 311;plot(signal_123_band1(1:200,:),'b-');
% subplot 312;plot(signal_123_band2(1:200,:),'k-');
% subplot 313;plot(signal_123_band3(1:200,:),'g-');
%% Step3: ¶ÔÊ±ÓòÖØ½¨µÄÐÅºÅÊ¹ÓÃ·Ç×ÔÉíÖÐÐÄÆµÂÊµÄ³ÉÐÎÂË²¨Æ÷¼ÆËãÐÇ×ùµã
% [cons_123_f1_f2] = cap_match_filter(signal_123_band1, filt2_I_path, filt2_Q_path,taps,upsampleno_mband);
% [cons_123_f2_f1] = cap_match_filter(signal_123_band2, filt1_I_path, filt1_Q_path,taps,upsampleno_mband);
% [cons_123_f2_f3] = cap_match_filter(signal_123_band2, filt3_I_path, filt3_Q_path,taps,upsampleno_mband);
% [cons_123_f3_f2] = cap_match_filter(signal_123_band3, filt2_I_path, filt2_Q_path,taps,upsampleno_mband);
cons_123_f1_f2 = CAPmatch_filter(signal_123_band1, gt1, t, fc(2), taps, upsampleno, 0);
cons_123_f2_f1 = CAPmatch_filter(signal_123_band2, gt1, t, fc(1), taps, upsampleno, 0);
cons_123_f2_f3 = CAPmatch_filter(signal_123_band2, gt1, t, fc(3), taps, upsampleno, 0);
cons_123_f3_f2 = CAPmatch_filter(signal_123_band3, gt1, t, fc(2), taps, upsampleno, 0);
cons_123_stage2 = [cons_123_f1_f2; cons_123_f2_f1; cons_123_f2_f3; cons_123_f3_f2];
%% Step 4:¶ÔÓÚStep3ÌáÈ¡µÄ½»µþ³É·ÖµÄÐÇ×ùµã¹¹³ÉÓëStep1»ñµÃµÄÈý¸ö´øµÄ¶ÀÁ¢³É·Ö¹¹³É¾ØÕó¼ÆËã¶ÀÁ¢³É·Ö
cons_123_be_dnn = [cons_123_stage1;cons_123_stage2];%7¸ö³É·Ö
%% ICA
cons_123_bf_ICA = [cons_123_stage1;cons_123_stage2];%13¸ö³É·Ö
[cons_123_af_ICA,W_output, counter, Q, W_iter] = cfastica_optimization(cons_123_bf_ICA,0,4);

abs_cons = (abs(cons_123_af_ICA)).';
abs_cons_123 = var(abs_cons);
[var_cons_dnn, index] = sort(abs_cons_123,'ascend');
% cons_rx_QPSK_no_order = [cons_12345_af_ICA(index(1),:);cons_12345_af_ICA(index(2),:);cons_12345_af_ICA(index(3),:);...
%     cons_12345_af_ICA(index(4),:);cons_12345_af_ICA(index(5),:);cons_12345_af_ICA(index(6),:);cons_12345_af_ICA(index(7),:)];
cons_rx_QPSK_no_order = [];
for IC_count = 1 : 7 %7:ÒªÈ¡µÄ¶ÀÁ¢³É·Ö
    cons_rx_QPSK_no_order = [cons_rx_QPSK_no_order;cons_123_af_ICA(index(IC_count),:)];
end
%ºóÃæ6£¬7ÅÅÃûµÄÒ²»­Ò»ÏÂ£¬¹Û²ìÊÇ·ñÑ¡¶ÔÁËÇ°Îå¸ö
% var_cons_dnn  % ¼ì²é·½²îµÄ·Ö²¼
% figure;
% suptitle('Step5 ¶þ·¶Êý£¬×¢ÒâcheckµÚÁùÕÅ£¬µÚÆßÕÅÍ¼');
% draw_len = 2000;
% subplot 241;plot(real(cons_rx_QPSK_no_order(1,1:draw_len)),imag(cons_rx_QPSK_no_order(1,1:draw_len)),'y.');
% subplot 242;plot(real(cons_rx_QPSK_no_order(2,1:draw_len)),imag(cons_rx_QPSK_no_order(2,1:draw_len)),'b.');
% subplot 243;plot(real(cons_rx_QPSK_no_order(3,1:draw_len)),imag(cons_rx_QPSK_no_order(3,1:draw_len)),'k.');
% subplot 244;plot(real(cons_rx_QPSK_no_order(4,1:draw_len)),imag(cons_rx_QPSK_no_order(4,1:draw_len)),'g.');
% subplot 245;plot(real(cons_rx_QPSK_no_order(5,1:draw_len)),imag(cons_rx_QPSK_no_order(5,1:draw_len)),'b.');
% subplot 246;plot(real(cons_rx_QPSK_no_order(6,1:draw_len)),imag(cons_rx_QPSK_no_order(6,1:draw_len)),'b.');
% subplot 247;plot(real(cons_rx_QPSK_no_order(7,1:draw_len)),imag(cons_rx_QPSK_no_order(7,1:draw_len)),'b.');
%% Step 6: ½«½ÓÊÕ¶ËÌáÈ¡µ½µÄÈýÂ·´øÏàÎ»Ðý×ªµÄÐÅºÅÆ¥Åäµ½·¢Éä¶ËµÄÈýÂ··¢ÉäÐÇ×ùµãÉÏ 
pilot_len = 2000;
tx_cons_pilot = [complex_data1(1:pilot_len).'; complex_data2(1:pilot_len).'; complex_data3(1:pilot_len).'];
rx_cons_pilot = cons_rx_QPSK_no_order;
[rec_and_phs_cons] = H_rec_phs_rotation_glb(tx_cons_pilot,rx_cons_pilot);
% figure;
% suptitle('Step6 Æ¥Åä£¬Ðý×ªÏàÎ»');
% subplot 221;plot(real(rec_and_phs_cons(1,:)),imag(rec_and_phs_cons(1,:)),'b.');
% subplot 222;plot(real(rec_and_phs_cons(2,:)),imag(rec_and_phs_cons(2,:)),'k.');
% subplot 223;plot(real(rec_and_phs_cons(3,:)),imag(rec_and_phs_cons(3,:)),'y.');
%% Step 8: ¼ÆËãBER
% Tx1 = data_mapping(1,:);
% Tx2 = data_mapping(2,:);
% Tx3 = data_mapping(3,:);
% Rx1 = cons_123_filt1;
% Rx2 = cons_123_filt2;
% Rx3 = cons_123_filt3;
% 1 : without BSS-ICA algorithm 
% [ber_raw(1),ser_raw(1)] = MonteCarlo_BER_Estimation(Tx1, Rx1, QAMorder(1));
% [ber_raw(2),ser_raw(2)] = MonteCarlo_BER_Estimation(Tx2, Rx2, QAMorder(2));
% [ber_raw(3),ser_raw(3)] = MonteCarlo_BER_Estimation(Tx3, Rx3, QAMorder(3));
% ber_raw_avg = mean(ber_raw);
Rx1 = rec_and_phs_cons(1,:);
Rx2 = rec_and_phs_cons(2,:);
Rx3 = rec_and_phs_cons(3,:);
% disp(['BER raw is: ',num2str(ber_raw_avg)]);
% 2: BSS-ICA algorithm without RLS...
[ber(1),ser(1)] = MonteCarlo_BER_Estimation(Tx1(start:end), Rx1(start:end), QAMorder);
[ber(2),ser(2)] = MonteCarlo_BER_Estimation(Tx2(start:end), Rx2(start:end), QAMorder);
[ber(3),ser(3)] = MonteCarlo_BER_Estimation(Tx3(start:end), Rx3(start:end), QAMorder);
ber_wo_RLS = mean(ber);
disp(['BER with BSS-ICA is: ',num2str(ber_wo_RLS)]);
% ´æÎÄ¼þ·ÖÎö
% CF_BER = [rolloff,cf,ber_raw_avg,ber_wo_RLS];%,ber_w_RLS,ber_ratio ±¸ÓÃ
% CF_BER = [SNR, rolloff, cf, eta, ber_raw_avg,ber_wo_RLS, ber_raw, ber];
% save('band3ÏêÏ¸·ÂÕæ.txt','CF_BER','-ascii','-append');
disp('The end...');
% 

% %% single-band
% upsampleno = round(fs/Rs);
% taps = 8 * upsampleno + 1;
% ts = linspace(1,numofsymbols*upsampleno,numofsymbols*upsampleno)-1-numofsymbols*upsampleno/2;
% t = ts/fs;
% QAMorder = M;
% % SRRC
% gtr1 = cos(pi*t*(1+rolloff)*Rs)+sin(pi*t*(1-rolloff)*Rs)./(4*rolloff*t*Rs);
% gtr2 = gtr1./(1-(4*rolloff*t*Rs).^2);
% gtr = gtr2*4*rolloff/pi*sqrt(Rs);
% gtr(numofsymbols*upsampleno/2+1) = sqrt(Rs)*(1+rolloff*(4/pi-1));
% gt2 = gtr;
% decimal_data = [decimal_data1; decimal_data2; decimal_data3];
% complex_data = [complex_data1; complex_data2; complex_data3];
% complex_data_for_nn = [real(complex_data), imag(complex_data)];
% save y1data_for_dnn.txt complex_data_for_nn -ascii
% cap_signal = CAPmod(complex_data, gt2, t, Bcap/2, taps, upsampleno);
% cap_signal = vlc_channel(cap_signal, SNR, 100, factor, 1);
% save('up1_data_for_dnn.txt', 'cap_signal', '-ascii');
% figure
% plot(20*log10(fftshift(abs(fft(cap_signal)))));
% [cons_1band, ~, ber_raw_1band] = CAPmatch_filter(cap_signal, gt2, t, Bcap/2, taps, upsampleno, 0, QAMorder, decimal_data);
% % [ber_raw_1band, ser_raw_1band] = MonteCarlo_BER_Estimation(complex_data(start:end), cons_1band(start:end), QAMorder);
% disp(['Single-band raw BER is: ',num2str(ber_raw_1band)]);
% addpath('D:\1ÊµÑéÊÒÎÄ¼þ\SCAP_DNN\SCAP_DNNpy2');
% predict_data = load('predict_data_for_1band.txt');
% Rx = predict_data(:, 1) + 1i*predict_data(:, 2);
% figure(100)
% subplot(224)
% plot(Rx, '.')
% [ber_1band, ser_1band] = MonteCarlo_BER_Estimation(complex_data(3*start-2:end), Rx, QAMorder);
% disp(['Single band BER with DNN is: ',num2str(ber_1band)]);

