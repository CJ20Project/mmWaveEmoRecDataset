function heartBeatPSDEnergy = PSDEnergy_ppg(ppgSmooth, winSize, winStride, fs)
    % Get the signal length
    N = length(ppgSmooth);
    
    % Calculate the number of valid windows (considering boundary conditions)
    numWindows = floor((N - winSize) / winStride) + 1;
    
    % Define the frequency band division points for heartbeat (unit: Hz)
    freSegHea = [1.0, 1.3, 1.6, 1.8];
    % Actual number of frequency bands = number of division points - 1
    numFreSegHea = numel(freSegHea) - 1;  
    
    % Pre-allocate the output matrix (frequency bands × time windows)
    heartBeatPSDEnergy = zeros(numFreSegHea, numWindows);
    
    % Main loop for sliding window processing
    for i = 1 : numWindows
        % Calculate the position of the current window
        startIndex = (i-1) * winStride + 1;
        endIndex = startIndex + winSize - 1;
        
        % Extract data for the current window
        windowHeartBeat = ppgSmooth(startIndex : endIndex);
        
        % Calculate Power Spectral Density (using pwelch method)
        % Note: Using default window, overlap, and NFFT settings for pwelch
        [pxxHea, fHea] = pwelch(windowHeartBeat, [], [], [], fs);
        
        % Calculate the energy in each frequency band
        for k = 1 : numFreSegHea
            % Calculate the average PSD energy in the [f1, f2] band
            f1 = freSegHea(k);
            f2 = freSegHea(k + 1);
            % The bandpower function returns the total power in the band. 
            % To get the average power spectral density, we divide by the bandwidth.
            heartBeatPSDEnergy(k, i) = bandpower(pxxHea, fHea, [f1, f2], 'psd') / (f2 - f1);
        end
    end
end