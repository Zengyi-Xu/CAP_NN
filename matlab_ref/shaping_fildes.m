function filter = shaping_fildes(rolloff, span, sps, shape)
if strncmp(shape, 'rc', 1)
    filter = rcosdesign(rolloff, span, sps, 'normal');
elseif strncmp(shape, 'srrc', 1)
    filter = rcosdesign(rolloff, span, sps, 'sqrt');
elseif strncmp(shape, 'btn', 1)
    delay = span*sps/2;
    t = (-delay:delay)/sps;
    filter = sinc(t).*(2*pi*rolloff*t/log(2).*sin(pi*rolloff*t) + ...
        2*cos(pi*rolloff*t)-1)./((pi*rolloff*t/log(2)).^2+1);
end
    