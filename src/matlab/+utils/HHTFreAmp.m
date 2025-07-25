function [meanFrequencies, meanSquaredAmplitudes] = HHTFreAmp(data, winSize, winStride, fs, firstN)
    % Parameter validation
    if firstN < 1
        error('The firstN parameter must be greater than or equal to 1');
    end

    % Get the signal length
    N = length(data);
    
    % Calculate the number of valid windows (considering boundary conditions)
    numWindows = floor((N - winSize) / winStride) + 1;
    
    % Pre-allocate output matrices (firstN × time windows)
    meanFrequencies = zeros(firstN, numWindows);
    meanSquaredAmplitudes = zeros(firstN, numWindows);
    
    % Main loop for sliding window processing
    for i = 1:numWindows
        % Calculate the position of the current window
        startIndex = (i-1)*winStride + 1;
        endIndex = startIndex + winSize - 1;
        
        % Extract data for the current window
        windowData = data(startIndex:endIndex);
        
        % EMD decomposition
        [imfs, ~] = emd(windowData, 'Interpolation', 'pchip', 'Display', 0);
        imfs = imfs';  % Transpose to have IMFs as row vectors
        
        % Get the actual number of IMFs (to handle cases where emd produces fewer than firstN IMFs)
        numActualIMFs = size(imfs, 1);
        numToProcess = min(firstN, numActualIMFs);
        
        % Loop for processing IMF components
        for j = 1:numToProcess
            % Calculate Hilbert statistics
            [meanFreq, meanAmp2] = HilbertStats(imfs(j,:), fs);
            
            % Store the results
            meanFrequencies(j,i) = meanFreq;
            meanSquaredAmplitudes(j,i) = meanAmp2;
        end
        
    end
end

%%
% Calculate statistical features of an IMF component using the Hilbert Transform
function [meanFreq, meanAmp2] = HilbertStats(imf, fs)
    % Time interval (seconds)
    dt = 1/fs;
    
    % Analytic signal
    analyticSignal = hilbert(imf);
    
    % Extract the envelope
    amplitude = abs(analyticSignal);
    
    % Unwrap the phase
    phase = unwrap(angle(analyticSignal));
    
    % Calculate the instantaneous frequency (Hz)
    % Note: Calculated via phase difference, padded with zero at the end to maintain length
    instantaneousFrequency = [diff(phase)/(2*pi*dt), 0];
    
    % Frequency correction: filter out non-physical negative frequencies
    instantaneousFrequency(instantaneousFrequency < 0) = 0;
    
    % Calculate statistical features
    meanFreq = mean(instantaneousFrequency);  % Mean frequency
    meanAmp2 = mean(amplitude.^2);           % Mean of the squared amplitude
end