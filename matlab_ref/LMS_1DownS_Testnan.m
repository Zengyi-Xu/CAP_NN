function [k,y,E,W] = LMS_1DownS_Testnan(taps_LMS,u_LMS,numofTs,Rxdata,Txdata)

Rxdata=reshape(Rxdata,1,[]).';   % received data
Rxdata=Rxdata/sqrt(mean(abs(Rxdata).^2));

Txdata=reshape(Txdata,1,[]).';    % original data
% Txdata=qammod(Txdata,2^Bitpersymbol);
Txdata=Txdata/sqrt(mean(abs(Txdata).^2));

x=Rxdata(1:numofTs);   % training symbol
d=Txdata(1:numofTs);
% d=upsample(d,2);

[Mx,Nx] = size(x);

Sx=[Mx,Nx];
ntr = max(Sx);              %  temporary number of iterations

compensation=taps_LMS;

y = zeros(length(Sx),1);              %  initialize output signal vector
E = zeros(length(Sx),1);              %  initialize error signal vector


W=zeros(1,taps_LMS);   % taps of LE
delta=0.0000001;
% V=delta*eye(taps_volterra);   % taps of volterra NLE
%%%%%%
%  Main loop
nn=1;
for i=1:1
    n=compensation;
   
    while n<=ntr
        
        X_LMS = x(n-compensation+(compensation+1)/2+(taps_LMS-1)/2:-1:n-compensation+(compensation+1)/2-(taps_LMS-1)/2);            %  LMS linear
        
%         X_volterra=x(n-compensation+(compensation+1)/2+(taps_volterra-1)/2:-1:n-compensation+(compensation+1)/2-(taps_volterra-1)/2);       % LMS nonlinear
%         X_v=triu(X_volterra*X_volterra');
%         VV=sum(sum(V.*X_v));
        
        y(nn) = W*X_LMS;             %  compute and assign current output signal sample
        e = d(n-compensation/2+1/2) - y(nn);     %  compute error
        
        W = W + u_LMS*e*X_LMS';   %  update filter coefficient vector
%         V=V+u_volterra*e*X_v;
        E(nn)=e;     % error
        
        n=n+1;

        nn=nn+1;
    end
end
% y=y/sqrt(mean(abs(y).^2));
% figure;bar(real(W));
% figure;bar(diag(V));

n=compensation;
mm=1;
while n<=length(Rxdata)
    R_LMS=Rxdata(n-compensation+(compensation+1)/2+(taps_LMS-1)/2:-1:n-compensation+(compensation+1)/2-(taps_LMS-1)/2);
%     R_volterra=Rxdata(n-compensation+(compensation+1)/2+(taps_volterra-1)/2:-1:n-compensation+(compensation+1)/2-(taps_volterra-1)/2);
%     R_v=triu(R_volterra*R_volterra');
%     
%     R_vv=sum(sum(V.*R_v));
    k(mm)=W*R_LMS;
    n=n+1;

    mm=mm+1;
end
k=[Rxdata(1:(compensation-1)/2).' k Rxdata(end-(compensation-1)/2+1:end).']; %²¹³¥³éÍ·ËðÊ§
% k=[Rxdata(1:(compensation-1)/2).' k Rxdata(end-(compensation+1)/2+2:end).']; %²¹³¥³éÍ·ËðÊ§ OFDMÐèÒª

k=k/sqrt(mean(abs(k).^2));   % output after LMS+volterra

% figure();
% plot(abs(E),'b-');title('error');
