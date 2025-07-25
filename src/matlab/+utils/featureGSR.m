function featureMatrix ...
    = featureGSR(data, winSize, winStride, fs)

N = length(data);
numWindows = floor((N - winSize) / winStride) + 1;

% Define a helper function to calculate time-domain statistical features
calculate_time_domain_stats = @(signal_in) [
    median(signal_in, 'omitnan'), mean(signal_in, 'omitnan'), std(signal_in, 'omitnan'), ...
    min(signal_in, [], 'omitnan'), max(signal_in, [], 'omitnan'), ...
    min(signal_in, [], 'omitnan') / (max(signal_in, [], 'omitnan') + eps('double'))
];

featureMatrix = zeros(numWindows, 24);

for i = 1 : numWindows
    startIndex = (i-1) * winStride + 1;
    endIndex = startIndex + winSize - 1;
    
    sc_signal = data(startIndex : endIndex);
    sc_diff1 = diff(sc_signal);
    sc_diff2 = diff(sc_signal, 2);
    
    all_features = [];

    % Time-domain features of the original SC signal
    if isempty(sc_signal) || all(isnan(sc_signal))
        features_sc_original = NaN(1,6); 
    else 
        features_sc_original = calculate_time_domain_stats(sc_signal); 
    end
    all_features = [all_features, features_sc_original];
    
    % Time-domain features of the 1st derivative of SC
    if isempty(sc_diff1) || all(isnan(sc_diff1))
        features_sc_diff1 = NaN(1,6); 
    else 
        features_sc_diff1 = calculate_time_domain_stats(sc_diff1); 
    end
    all_features = [all_features, features_sc_diff1];
    
    % Time-domain features of the 2nd derivative of SC
    if isempty(sc_diff2) || all(isnan(sc_diff2))
        features_sc_diff2 = NaN(1,6); 
    else 
        features_sc_diff2 = calculate_time_domain_stats(sc_diff2); 
    end
    all_features = [all_features, features_sc_diff2];
    
    % Extract PSD features using Welch's method (0-2 Hz, based on the conductance signal)
    window_duration_sec = 5;
    window_samples = floor(window_duration_sec * fs);
    sc_signal_for_pwelch = sc_signal;
    
    if length(sc_signal_for_pwelch) < window_samples && length(sc_signal_for_pwelch) > 0
        window_samples = length(sc_signal_for_pwelch);
        disp('Welch window length adjusted to signal length because the signal is too short.');
    elseif length(sc_signal_for_pwelch) == 0 || window_samples == 0 || all(isnan(sc_signal_for_pwelch))
        warning('Effective length of the conductance signal is 0 or all NaN, cannot compute PSD features. Setting to NaN.');
        features_psd_sc = NaN(1,6);
        all_features = [all_features, features_psd_sc];
    else
        input_to_pwelch = sc_signal_for_pwelch;
        if any(isnan(input_to_pwelch))
            warning('Signal for pwelch still contains NaN values. Attempting to handle with fillmissing.');
            input_to_pwelch = fillmissing(input_to_pwelch, 'previous', 'EndValues','nearest');
            input_to_pwelch = fillmissing(input_to_pwelch, 'next', 'EndValues','nearest');
            
            if any(isnan(input_to_pwelch)) || length(input_to_pwelch) < window_samples
                warning('Could not effectively handle NaNs in pwelch input or the resulting length is insufficient. PSD features will be set to NaN.');
                features_psd_sc = NaN(1,6);
                all_features = [all_features, features_psd_sc];
                input_to_pwelch = []; % Mark as not usable for pwelch
            end
        end
        
        if ~isempty(input_to_pwelch) % Proceed only if input_to_pwelch is valid
            overlap_percent = 0.5;
            noverlap_samples = floor(window_samples * overlap_percent);
            nfft = max(256, 2^nextpow2(window_samples));
            
            try
                [Pxx, F_psd] = pwelch(input_to_pwelch, hamming(window_samples), noverlap_samples, nfft, fs);
                freq_range_indices = (F_psd >= 0 & F_psd <= 2);
                Pxx_selected = Pxx(freq_range_indices);
                
                if isempty(Pxx_selected) || all(isnan(Pxx_selected))
                    warning('No valid PSD data points for the SC signal in the 0-2Hz range. PSD features will be set to NaN.');
                    features_psd_sc = NaN(1,6);
                else
                    psd_median = median(Pxx_selected, 'omitnan'); 
                    psd_mean = mean(Pxx_selected, 'omitnan');
                    psd_std = std(Pxx_selected, 'omitnan'); 
                    psd_max = max(Pxx_selected, [], 'omitnan');
                    psd_min = min(Pxx_selected, [], 'omitnan');
                    
                    if isempty(psd_max) || isempty(psd_min) || isnan(psd_max) || isnan(psd_min)
                        psd_range = NaN; 
                    else 
                        psd_range = psd_max - psd_min; 
                    end
                    features_psd_sc = [psd_median, psd_mean, psd_std, psd_max, psd_min, psd_range];
                end
            catch pwelch_ME
                warning('Error during pwelch calculation. PSD features will be set to NaN.');
                features_psd_sc = NaN(1,6);
            end
            all_features = [all_features, features_psd_sc];
        end
    end
    
    featureMatrix(i, :) = all_features;
end
end