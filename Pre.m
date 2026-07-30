% modify by JY
% 20211108
% modify by JY for DMT
% 20211116

% figure;plot(abs(ha));
% temp=abs(ha);
% save ha4.txt temp -ascii;


flag_pre_method=5; % 0 = normal; 1 = inverse+normal; 2 = cut off; 3 = peak point; 4 = peak point fit; 5 = Hardware Pre
equal_dB=20;
equal_dB2=20;

load ha2.txt;
load ha3.txt
load ha4.txt
temp=ha2+ha3+ha4;
figure;plot(temp);
temp_inverse=1./temp;
temp_inverse=20*log10(temp_inverse);
temp_inverse=temp_inverse-temp_inverse(1);
figure;plot(temp_inverse);

switch flag_pre_method
    case 0
        fitresult = createFit(temp);%%cftool best sqrt
        %         fitresult = createFit_VLC(temp);%%
        temp2=fitresult(1:length(temp));
        temp3=(temp2).^(1/2);
        
    case 1
        temp_inverse=1./temp;
        temp_inverse=20*log10(temp_inverse);
        temp_inverse=temp_inverse-temp_inverse(end);
        fitresult = createFit_VLC_inverse(temp_inverse);%%
        temp2=fitresult(1:length(temp_inverse));
        temp2=1./(10.^(temp2/20));
        temp3=sqrt(abs(temp2));
    case 2
        temp_inverse=1./temp;
        temp_inverse=20*log10(temp_inverse);
        temp_inverse=temp_inverse-temp_inverse(1);
        temp_index=find(temp_inverse<-equal_dB);
        count_index=temp_index(1)+1;
        temp_fit=temp(1:count_index);
        figure;plot(temp_inverse)
        %         fitresult = createFit(temp_fit);%%cftool best sqrt
        fitresult = createFit_VLC(temp_fit);%%
        temp2=fitresult(1:length(temp_fit));
        temp3=sqrt(abs(temp2));
        temp3=[temp3;ones(length(temp)-count_index,1)*min(temp3)];
    case 3
        temp_inverse=1./temp;
        temp_inverse=20*log10(temp_inverse);
        temp_inverse=temp_inverse-temp_inverse(1);
        temp_index=find(temp_inverse<-equal_dB);
        count_index=temp_index(1)+1;
        fitresult = createFit_VLC_inverse(temp_inverse);%%
        temp2=fitresult(1:length(temp_inverse));
        temp2=1./(10.^(temp2/20));
        temp2=temp2./temp2(count_index);
        temp2_hig=temp2(count_index:end);
        temp2_low=temp2(1:count_index-1);
        temp3=[(temp2_low).^(5/8);(temp2_hig).^(1/8);];
    case 4
        temp_inverse=1./temp;
        temp_inverse=20*log10(temp_inverse);
        temp_inverse=temp_inverse-temp_inverse(1);
        fitresult = createFit_VLC_inverse(temp_inverse);%%
        temp2=fitresult(1:length(temp_inverse));
        temp_index=find(temp2<-equal_dB);
        count_index=temp_index(1)+1;
        temp_index2=find(temp2<-equal_dB2);
        count_index2=temp_index2(1)+1;
        temp2=1./(10.^(temp2/20));
        temp2=temp2./temp2(count_index);
        temp2_hig=temp2(count_index:end);
        temp2_low=temp2(1:count_index-1);
        temp3=[(temp2_low).^(6/8);(temp2_hig).^(1/100)];
    case 5
%         load Hardware_pre.mat
%         Fbegin= Hardware_pre(1);
%         AdbV  = Hardware_pre(2);
%         FcenV = Hardware_pre(3); %MHz
%         FhalfV= Hardware_pre(4); %MHz
%         Fend  = Hardware_pre(5);
    
        % all the params are norm. to Fend
        Fbegin=1;
        AdbV=25;
        FcenV=600;  % <1000 
        FhalfV=400; % <1000
        Fend=600;

%         Fbegin=1;
%         AdbV=10;
%         FcenV=450; %MHz
%         FhalfV=80; %MHz
%         Fend=600;

        R0=50;
        FcenV=FcenV*1E6;
        FhalfV=FhalfV*1E6;
        [C11,L11,R11,C22,L22,R22]=BridgeT_II_1(FcenV,FhalfV,AdbV,R0);
        L11_H=L11*1E-9;  %Use H as the Unit
        C11_F=C11*1E-12; %Use F as the Unit
        f=1E6:1E6:1E9;  %1M-1G response
%         f=linspace(1E5,1E9,217824);  %1M-1G response
        w=2*pi*f;
        % x11_up=1i*w*L11_H;
        % x11_down=1-L11_H*C11_F*(w.^2);
        % x11=x11_up./x11_down;
        x11=1i*w*L11_H+1./(1i*w*C11_F);
        r11=R11;
        z11_up=r11.*x11;
        z11_down=r11+x11;
        z11=z11_up./z11_down;
        b_line=(1+z11/R0);
        b_line=b_line.^2;
        b_line=abs(b_line);
        %b_line=b_line.^2;
        b_log=10*log10(b_line);
        b_log=-b_log;
        % figure(1);
        f_dis=f/1E6;
        plot(f_dis,b_log,'r');
        grid on;
        xlabel('MHz')
        ylabel('dB');
        f_use=(b_log(Fbegin:Fbegin+Fend));
        index_f_use=1:length(f_use);
        index_f_use_right=linspace(1,length(f_use),length(temp));
        f_use_right=spline(index_f_use,f_use,index_f_use_right);
%         f_use_right=resample(f_use,length(temp),length(f_use));
        temp3=(10.^(f_use_right/20));
        temp3=reshape(temp3,[],1);
        save f_grid.txt f_dis -ascii
        save f_hardware_dB.txt b_log -ascii;
end
temp3_inverse_dB=20*log10(temp3);
% temp3_inverse_dB(end)
% temp3_inverse_dB(count_index2)
figure;plot(temp3_inverse_dB);

save th7.txt temp3 -ascii