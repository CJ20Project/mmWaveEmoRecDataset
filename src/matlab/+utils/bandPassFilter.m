function y = bandPassFilter(x, Fs, f1, f2, n_total)
    % Input validation
    if f1 <= 0 || f2 >= Fs/2 || f1 >= f2
        error('Invalid frequency range: 0 < f1 < f2 < Fs/2');
    end
    if mod(n_total,2) ~= 0
        error('Filter order must be even');
    end

    % Normalize cutoff frequencies
    Wn = [f1, f2] / (Fs/2);

    % Design a Butterworth bandpass filter
    n = n_total / 2;
    [b, a] = butter(n, Wn, 'bandpass');
    
    % Apply zero-phase filtering
    y = filtfilt(b, a, x);
end