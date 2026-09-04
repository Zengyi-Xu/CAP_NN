% P³½±àÐ´µÄcap_gen code,ÓÅ»¯ÕûºÏ³Éº¯Êý£¬±ãÓÚµ÷ÊÔ
% ²ÎÊýÊäÈë¸Ä±äÁË£¬Ö±½ÓÊäÈëµÄÊÇÃ¿¸ö×Ó´øµÄÖÐÐÄÆµÂÊ£¬¶ø²»ÊÇ¹éÒ»»¯µÄÏµÊý
function [data_aftermap,data, normal_data, filter_I_path, filter_Q_path,var_signal] = cap_gen(QAMorder,...
    SymbolRate, numofsymbols, upsampleno,taps,alpha,subcar, rng_num, In_dataAftermap)

%% PRBS signal sequence
disp('PRBS generation...')
rng(rng_num);
data=randi(QAMorder,numofsymbols,1)-1;
%% QAM mapping
disp('Mapping...')
data_aftermap=qammod(data,QAMorder);
% figure;
% plot(real(data_aftermap),imag(data_aftermap),'.b');
% title('Tx Constellation');xlabel('Inphase');ylabel('Quadrature');
% Èç¹ûÊäÈë²ÎÊýÎª°Ë¸ö£¬¾Í°Ñ×îºóÒ»¸ö²ÎÊýÄÃÀ´½øÐÐCAP modulation
if (nargin == 9)
    data_aftermap = In_dataAftermap;
end
%% Upsample
disp('upsampling...')
data_afterUpsample = upsample(data_aftermap,upsampleno);
%% Generate shaping filter
disp('SRC filter generation...')
% SymbolRate  =100e6; %ÉÏ²ÉÑùºóµÄ1/Ts
samplerate  = upsampleno*SymbolRate;%·¢Éä¶ËAWGµÄ²ÉÑùÂÊ
delta_t     =1/samplerate;%ÉÏ²ÉÑùºóµÄTs
f           =SymbolRate;

upsamplesymbol=numofsymbols*upsampleno;%ÉÏ²ÉÑùºóµÄ·ûºÅÊý
t_sequence_norm=linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2;
t_sequence=t_sequence_norm*delta_t;
% SRRCÂË²¨Æ÷:¾­¹ý±È¶Ô£¬ÓëPaul HaighÌá³öµÄ´«Í³gtr°æ±¾Ò»ÖÂ
gtr1    =cos(pi*t_sequence*(1+alpha)*f)+sin(pi*t_sequence*(1-alpha)*f)./(4*alpha*t_sequence*f);
gtr2    =gtr1./(1-(4*alpha*t_sequence*f).^2);
gtr     =gtr2*4*alpha/pi;%*sqrt(f);

% Paul HaighÂË²¨Æ÷:<Analysis of Nyquist Pulse Shapes for CAP Modulation in VLC> ÎÄÖÐµÄBTNÂË²¨Æ÷¶Ô
% gtr1 = sinc(f*t_sequence).*(2*pi*alpha*t_sequence*f/log(2).*sin(pi*alpha*t_sequence*f)+2*cos(pi*alpha*t_sequence*f)-1);
% gtr2 = (pi*alpha*t_sequence*f/log(2)).^2+1; 
% gtr = gtr1./gtr2;

gtr(upsamplesymbol/2+1)=(1+alpha*(4/pi-1));%*sqrt(f);
filter_I_path=reshape(gtr.*cos(2*pi*subcar*t_sequence),[],1);
filter_Q_path=reshape(gtr.*sin(2*pi*subcar*t_sequence),[],1);

%% pulse shaping
disp('pulse shaping...')

gtI1 = filter_I_path;
gtQ1 = filter_Q_path;
Idata1 = real(data_afterUpsample);
Idata1 = reshape(Idata1,[],1);
Qdata1 = imag(data_afterUpsample);
Qdata1 = reshape(Qdata1,[],1);

filterI = gtI1((1:taps)+numofsymbols*upsampleno/2-(taps-1)/2);
filterQ = gtQ1((1:taps)+numofsymbols*upsampleno/2-(taps-1)/2);
% figure;plot(filterI,'r-.');hold on; plot(filterQ,'b-.')
dataI = [Idata1(end-(taps-1)/2+1:end);Idata1;Idata1(1:(taps-1)/2)];
dataQ = [Qdata1(end-(taps-1)/2+1:end);Qdata1;Qdata1(1:(taps-1)/2)];
DataCapI = conv(dataI,filterI,'valid');
DataCapQ = conv(dataQ,filterQ,'valid');
output_data = DataCapI-DataCapQ;
%% Average Power Normalization
disp('normalization...')
data_n = reshape(output_data,[],1);    
 var_signal = sqrt(mean(abs(data_n).^2));
normal_data = data_n./var_signal;%¹¦ÂÊ¹éÒ»»¯
% normal_data = data_n;%¹¦ÂÊ²»×ö¹éÒ»»¯
end