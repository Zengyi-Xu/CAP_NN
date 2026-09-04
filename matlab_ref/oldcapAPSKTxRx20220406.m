clear all;
close all;
%%
% SingleBand CAP Modulation simulation in VLC system.
constellation = 'QAM';
%could be QAM, APSK
%% system parameter
numofsymbols=51200;
QAMorder=32;
upsampleno=4;
alpha=0.205;
subcar=1/2;
startFreq=1/100;
% BW=1+alpha; %¿É±äspacing
% subcar=BW/2+startFreq;
%% PRBS signal sequence
rng(100);
DECdata=randi(QAMorder,numofsymbols,1)-1;
namepre = 'decimal_';
namepost='.txt';
name1=[namepre,num2str(QAMorder),namepost];
eval(['save ' name1 '  DECdata -ASCII']);
%% QAM mapping
% qamdata1=qammod(data,QAMorder);
qamdata1=GS_CCSDSmodulation_cons(length(DECdata),DECdata,QAMorder,constellation);
data_afterUpsample = upsample(qamdata1,upsampleno);

% figure;
% plot(real(qamdata1),imag(qamdata1),'.b');
% title('Tx Constellation');xlabel('Inphase');ylabel('Quadrature')

%% Generate shaping filter
% disp('SRC filter generation...')
upsamplesymbol=numofsymbols*upsampleno;%ÉÏ²ÉÑùºóµÄ·ûºÅÊý
t_sequence_norm=linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2;
gtr1    =cos(pi*t_sequence_norm*(1+alpha)*0.25)+sin(pi*t_sequence_norm*(1-alpha)*0.25)./(4*alpha*t_sequence_norm*0.25);
gtr2    =gtr1./(1-(4*alpha*t_sequence_norm*0.25).^2);
gtr     =gtr2*4*alpha/pi;%*sqrt(f);
gtr(upsamplesymbol/2+1)=(1+alpha*(4/pi-1));%*sqrt(f);
BW=1+alpha;
subcar1=BW*subcar+startFreq;
filter_I_path=reshape(gtr.*cos(2*pi*subcar1*t_sequence_norm*0.25),[],1);
filter_Q_path=reshape(gtr.*sin(2*pi*subcar1*t_sequence_norm*0.25),[],1);

%% pulse shaping
% disp('pulse shaping...')
taps = 35; %  filter length (should be odd number !!) 

gtI1 = filter_I_path;
gtQ1 = filter_Q_path;
Idata1 = real(data_afterUpsample);
Idata1 = reshape(Idata1,[],1);
Qdata1 = imag(data_afterUpsample);
Qdata1 = reshape(Qdata1,[],1);

filterI = gtI1((1:taps)+numofsymbols*upsampleno/2-(taps-1)/2);
filterQ = gtQ1((1:taps)+numofsymbols*upsampleno/2-(taps-1)/2);

filterI = Pulse_shaping_ZY(alpha,taps,subcar,startFreq,upsampleno,numofsymbols,"I");
filterQ = Pulse_shaping_ZY(alpha,taps,subcar,startFreq,upsampleno,numofsymbols,"Q");

figure;plot(filterI,'r-.');hold on; plot(filterQ,'b-.')
DataCapI = conv(Idata1,filterI,'same');
DataCapQ = conv(Qdata1,filterQ,'same');
output_data = DataCapI-DataCapQ;
%% Average Power Normalization
% disp('normalization...')
dataout=output_data/sqrt(mean(abs(output_data).^2));
dataout=reshape(dataout,[],1);

namepre = 'data';
namepost='.txt';
name2 = [namepre,num2str(QAMorder),constellation];
name1=[namepre,num2str(QAMorder),constellation, namepost];
eval(['save ' name1 '  dataout -ASCII']);
% eval(['disp(''File Name is  ' name1 '.txt '')']);

makewfm(dataout,[name2,'.wfm'])    % æ³¢ä¿¡å·æ°æ®åå?
dataout=reshape(dataout,1,[]);
awg_SampleRate = 2e9;
xfre=((0:1:length(dataout)-1)-length(dataout)/2)*awg_SampleRate/1e6/length(dataout);
DataCap_f=fftshift(fft(dataout));
figure;plot(xfre,20*log10(abs(DataCap_f)),'b-')
title('Frequency Response');xlabel('Bandwidth(MHz)');ylabel('Amplitude(dB)')
%% Channel Simulation
% disp('Through Channel...')

SNR =27;% dB
Fs=100;
factor = 40;
n=1:length(dataout)/2;
fs=2*Fs/(length(dataout));
fs=fs*n;
ch1=exp(-fs/factor);
ch2=fliplr(ch1);
ch=[ch1 ch2];
data_xindao_fft=fft(dataout);
data_xindao_fft=data_xindao_fft.*ch;
datarx1=ifft(data_xindao_fft);
datarx1=awgn(datarx1,SNR,'measured');
datarx1 = reshape(datarx1,[],1);
%% waveform LMS volterra
%  disp('Begin LMS Volterra...');
%  for taps_LMS=35:2:131
% fprintf("taps: %d: ", taps_LMS);
% for u_LMS = 0.01:0.01:0.2
% fprintf("u: %f: ", u_LMS);
%  for taps_volterra=1:2:13
% fprintf("taps: %d: ", taps_volterra);
% for u_volterra = 0.0001:0.0001:0.001
% fprintf("u: %f: ", u_volterra);
taps_LMS=19;
u_LMS=0.015;
taps_volterra=11;
u_volterra=0.0004;
numofTs=8000;
[data_AfRLS,data_Ts,e,W]=LMS_volterra_1DownS_Testnan(taps_LMS,u_LMS,taps_volterra,u_volterra,numofTs,datarx1,dataout);
%% Matched Filtering
% disp('Matched Filtering ...')
offsetsample=0;
gtI = filter_I_path;
gtQ = filter_Q_path;
N = length(datarx1);
filterI = gtI((1:taps)+N/2-(taps-1)/2);
filterQ = gtQ((1:taps)+N/2-(taps-1)/2);
figure;plot(filterI,'r-.');hold on; plot(filterQ,'b-.')
% dataI = [received_data(end-(taps-1)/2+1:end);received_data;received_data(1:(taps-1)/2)];
% dataQ = [received_data(end-(taps-1)/2+1:end);received_data;received_data(1:(taps-1)/2)];
DataCapI = conv(datarx1,filterI,'same');
DataCapQ = conv(datarx1,filterQ,'same');
DataCap=DataCapI+1i*DataCapQ;
MatchDefilter_data=downsample(DataCap,upsampleno,offsetsample);
%%
a=0:QAMorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,QAMorder,constellation);
avp=sqrt(mean(abs(b).^2));
%% Post Equalization
% disp('LMS Equalization...')
taps_LMS=17;
u_LMS=0.005;
numofTs=8000;
  [data_AfRLS,data_Ts,e,W]=LMS_1DownS_Testnan(taps_LMS,u_LMS,numofTs,MatchDefilter_data,qamdata1);
recoverdata=data_AfRLS;
% recoverdata=data_recover1r1;
recoverdata=recoverdata./sqrt(mean(abs(recoverdata).^2))*avp;
recoverdata=reshape(recoverdata,1,[]);

origindata=qamdata1;
DECdata=reshape(DECdata,1,[]);
recoverdata=reshape(recoverdata,1,[]);
plot_hist3nan(recoverdata(taps_LMS:end-taps_LMS));
axis([-ceil(QAMorder.^0.5),ceil(QAMorder.^0.5),-ceil(QAMorder.^0.5),ceil(QAMorder.^0.5)]);

%%
% rxdata_dec=qamdemod(recoverdata,QAMorder);% ¼ÆËãBER
rxdata_dec=GS_CCSDSdemodulation_cons(length(recoverdata),recoverdata,QAMorder,constellation);
[symnum,ser]=symerr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
[bitnum,ber]=biterr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
fprintf('SER  BER\t%.4e    \t%.4e\n', ser, ber);
