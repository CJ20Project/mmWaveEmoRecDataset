function [fused_phase] = extractPhase(adcdata, RangFFT)
    %% Range-FFT
    fft_data = fft(adcdata, RangFFT);
    fft_data = fft_data.';
    
    %% Moving Target Indication (MTI) to remove static clutter
    data_mti = fft_data - mean(fft_data, 1);
    
    %% Multi-bin Fusion
    % Select the top K range bins with the strongest amplitudes
    K = 5;
    amp = mean(abs(data_mti), 1);
    [~, idx] = maxk(amp, K);
    
    % Phase Fusion
    phase_all = zeros(size(data_mti,1), K);
    for i = 1:K
        sig = data_mti(:, idx(i));
        phase_all(:,i) = unwrap(angle(sig));
    end
    
    % Calculate the weighted average of the phases (using amplitude as weights)
    weights = amp(idx);
    weights = weights / sum(weights);
    fused_phase = phase_all * weights';
end