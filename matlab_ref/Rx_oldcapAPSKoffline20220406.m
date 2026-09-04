close all;
clear all ;
%%
constellation = 'QAM';
%could be QAM, APSK
%     index1 = 458;
%     name1 = ['OSC_QAM_',num2str(index1), '_',num2str(constellation),'.txt'];
%     dataRx = importdata(name1);
datarx=importdata('seedOSC_oldcap64QAM275.txt');
dataout= importdata('data64QAM.txt');


%% 
numofsymbols=51200;
QAMorder=64;
upsampleno=4;
alpha=0.205;
subcar=1/2;
startFreq=1/100;

%%
dataout = reshape(dataout,1,[]);
dataout=dataout/sqrt(mean(abs(dataout).^2));
signal_in=dataout/max(abs(dataout));   % 
datarx = reshape(datarx,1,[]);
datarx=datarx/sqrt(mean(abs(datarx).^2));
signal_out=datarx/max(abs(datarx));          % 
figure;
plot(signal_in,signal_out,'b.');     %
% x1=-1:0.001:1;
% y1=-1:0.001:1;
x1=-1:0.001:1;
y1=-1:0.001:1;
hold on; plot(x1,y1,'r.');hold on;title('AM-AM');

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
taps_LMS=51;
u_LMS=0.003;
taps_volterra=1;
u_volterra=0.000005;
numofTs=8000;
[data_AfRLS,data_Ts,e,W]=LMS_volterra_1DownS_Testnan(taps_LMS,u_LMS,taps_volterra,u_volterra,numofTs,datarx,dataout);
%% Generate shaping filter
% disp('SRC filter generation...')
upsamplesymbol=numofsymbols*upsampleno;%ÉÏ²ÉÑùºóµÄ·ûºÅÊý
t_sequence_norm=linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2;
gtr1    =cos(pi*t_sequence_norm*(1+alpha)*0.25)+sin(pi*t_sequence_norm*(1-alpha)*0.25)./(4*alpha*t_sequence_norm*0.25);
gtr2    =gtr1./(1-(4*alpha*t_sequence_norm*0.25).^2);
gtr     =gtr2*4*alpha/pi;%*sqrt(f);
gtr(upsamplesymbol/2+1)=(1+alpha*(4/pi-1));%*sqrt(f);
% startFreq=1/30;
% BW=1+alpha; %¿É±äspacing
% subcar=BW/2+startFreq;
BW=1+alpha;
subcar1=BW*subcar+startFreq;
filter_I_path=reshape(gtr.*cos(2*pi*subcar1*t_sequence_norm*0.25),[],1);
filter_Q_path=reshape(gtr.*sin(2*pi*subcar1*t_sequence_norm*0.25),[],1);
%% Matched Filtering
% disp('Matched Filtering ...')
taps = 35; %  filter length (should be odd number !!) 
offsetsample=0;
gtI = filter_I_path;
gtQ = filter_Q_path;
N = length(data_AfRLS);
filterI = gtI((1:taps)+N/2-(taps-1)/2);
filterQ = gtQ((1:taps)+N/2-(taps-1)/2);
figure;plot(filterI,'r-.');hold on; plot(filterQ,'b-.')
% dataI = [received_data(end-(taps-1)/2+1:end);received_data;received_data(1:(taps-1)/2)];
% dataQ = [received_data(end-(taps-1)/2+1:end);received_data;received_data(1:(taps-1)/2)];
DataCapI = conv(data_AfRLS,filterI,'same');
DataCapQ = conv(data_AfRLS,filterQ,'same');
DataCap=DataCapI+1i*DataCapQ;
MatchDefilter_data=downsample(DataCap,upsampleno,offsetsample);
%%
a=0:QAMorder-1;
b=GS_CCSDSmodulation_cons(length(a),a,QAMorder,constellation);
avp=sqrt(mean(abs(b).^2));
%% Post Equalization
namepre = 'decimal_';
namepost='.txt';
name2=[namepre,num2str(QAMorder),namepost];
DECdata=importdata(name2);
qamdata1 = GS_CCSDSmodulation_cons(length(DECdata),DECdata,QAMorder,constellation);
% disp('LMS Equalization...')
taps_LMS=31;
u_LMS=0.0016;
numofTs=8000;
  [data_AfRLS,data_Ts,e,W]=LMS_1DownS_Testnan(taps_LMS,u_LMS,numofTs,MatchDefilter_data,qamdata1);
recoverdata=data_AfRLS;
% recoverdata=data_recover1r1;
recoverdata=recoverdata./sqrt(mean(abs(recoverdata).^2))*avp;
recoverdata=reshape(recoverdata,1,[]);

origindata=qamdata1;
DECdata=reshape(DECdata,1,[]);
recoverdata=reshape(recoverdata,1,[]);
figure;plot(recoverdata(taps_LMS:end-taps_LMS),'.');
gscatter(real(recoverdata),imag(recoverdata),DECdata);hold on

%%
rxdata_dec=GS_CCSDSdemodulation_cons(length(recoverdata),recoverdata,QAMorder,constellation);
error=recoverdata(rxdata_dec~=DECdata);
plot(error,'k*')
[symnum,ser]=symerr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
[bitnum,ber]=biterr(rxdata_dec(taps_LMS:end-taps_LMS),DECdata(taps_LMS:end-taps_LMS));
fprintf('SER  BER\t%.4e    \t%.4e\n', ser, ber);
