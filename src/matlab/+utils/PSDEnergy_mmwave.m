function [respirationPSDEnergy, heartBeatPSDEnergy] ...
    = PSDEnergy_mmwave(respiration, heartBeat, winSize, winStride, fs)

N = length(heartBeat);
numWindows = floor((N - winSize) / winStride) + 1;

% Define frequency band divisions for respiration and heartbeat signals
% Respiration signal frequency bands (unit: Hz)
freSegRes = [0, 0.1, 0.2, 0.3, 0.4];
numFreSegRes = numel(freSegRes) - 1;   % Number of frequency bands

% Heartbeat signal frequency bands (unit: Hz)
freSegHea = [1.0, 1.3, 1.6, 1.8];
numFreSegHea = numel(freSegHea) - 1;   % Number of frequency bands

% Initialize output matrices
respirationPSDEnergy = zeros(numFreSegRes, numWindows);
heartBeatPSDEnergy   = zeros(numFreSegHea, numWindows);

% Sliding window processing
for i = 1 : numWindows
    % Calculate start and end indices for the current window
    startIndex = (i-1) * winStride + 1;
    endIndex = startIndex + winSize - 1;
    
    % Extract signals for the current window
    windowRespiration = respiration(startIndex : endIndex);
    windowHeartBeat = heartBeat(startIndex : endIndex);
    
    % Calculate Power Spectral Density (PSD) for the respiration signal
    [pxxRes, fRes] = pwelch(windowRespiration, [], [], [], fs);
    
    % Calculate the average PSD energy for each band
    for j = 1 : numFreSegRes
        respirationPSDEnergy(j, i) = ...
            bandpower(pxxRes, fRes, [freSegRes(j), freSegRes(j + 1)], 'psd') / ...
            (freSegRes(j + 1) - freSegRes(j));
    end
    
    % Calculate Power Spectral Density (PSD) for the heartbeat signal
    [pxxHea, fHea] = pwelch(windowHeartBeat, [], [], [], fs);
    
    % Calculate the average PSD energy for each band
    for k = 1 : numFreSegHea
        heartBeatPSDEnergy(k, i) = ...
            bandpower(pxxHea, fHea, [freSegHea(k), freSegHea(k + 1)], 'psd') / ...
            (freSegHea(k + 1) - freSegHea(k));
    end
end
end
