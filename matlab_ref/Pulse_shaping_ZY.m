function filter = Gen_CAP_filters_ZY(alpha,taps,subcar,startFreq, upsampleno, numofsymbols,I_or_Q)

upsamplesymbol = upsampleno * numofsymbols;
t_sequence_norm=linspace(1,upsamplesymbol,upsamplesymbol)-1-upsamplesymbol/2;
gtr1    =cos(pi*t_sequence_norm*(1+alpha)*0.25)+sin(pi*t_sequence_norm*(1-alpha)*0.25)./(4*alpha*t_sequence_norm*0.25);
gtr2    =gtr1./(1-(4*alpha*t_sequence_norm*0.25).^2);
gtr     =gtr2*4*alpha/pi;%*sqrt(f);
gtr(upsamplesymbol/2+1)=(1+alpha*(4/pi-1));%*sqrt(f);
BW=1+alpha;
subcar1=BW*subcar+startFreq;

if I_or_Q == "I" % I channel
    filter = reshape(gtr.*cos(2*pi*subcar1*t_sequence_norm*0.25),[],1);
else
    filter = reshape(gtr.*sin(2*pi*subcar1*t_sequence_norm*0.25),[],1);
end
% truncation
filter = filter((1:taps)+numofsymbols*upsampleno/2-(taps-1)/2);
end