function [timeDomain, nonlinear] = ...
    HRVfeature_ppg(data, winSize, winStride, fs)

% Initialize empty structures with array fields
timeDomain = struct('meanNN', [], 'medianNN', [], 'SDNN', [], ...
                    'RMSSD', [], 'PNN50', [], ...
                    'meanRate', [], 'sdRate', [], 'HRVTi', []);
nonlinear = struct('PoincareSD1', []);

% Calculate the number of windows
numWindows = floor((length(data) - winSize) / winStride) + 1;

for i = 1:numWindows
    % Get the data for the current window
    startIdx = (i-1) * winStride + 1;
    endIdx = startIdx + winSize - 1;
    current_data = data(startIdx : endIdx);
    
    % Find peaks in the PPG signal
     [~, locs] = ...
         findpeaks(current_data, 'MinPeakHeight', 0.2*max(current_data), 'MinPeakDistance', fs*0.5);
         
    % Calculate NN intervals (unit: seconds)
    interval = diff(locs) / fs;  % Convert the interval between adjacent peaks from samples to time
    
    % Convert to milliseconds
    interval = interval * 1000;  % seconds -> milliseconds
    
    %% Time Domain Features
    meanNN = mean(interval);
    medianNN = median(interval);
    SDNN = std(interval);
    
    % Calculate RMSSD and pNN50 (requires at least two intervals)
    if numel(interval) >= 2
        diffs = diff(interval);            % Difference between adjacent intervals
        RMSSD = sqrt(mean(diffs.^2));      % Root mean square of successive differences
        abs_diffs = abs(diffs);            % Take the absolute value
        count = sum(abs_diffs > 50);       % Count the number of differences > 50ms
        PNN50 = count / numel(interval);   % The proportion
    else
        RMSSD = NaN; % If there are fewer than two intervals, return NaN
        PNN50 = NaN; % If there are fewer than two intervals, return NaN
    end
    
    % Calculate heart rate in beats per minute (BPM)
    rate = (60*1000) ./ interval; % bpm as unit
    meanRate = mean(rate);
    sdRate = std(rate);
    
    % HRV Triangular Index (HRVTi)
    binWidth = 7.8125; % unit: milliseconds
    [counts, ~] = histcounts(interval, 'BinWidth', binWidth);
    maxCount = max(counts);
    HRVTi = numel(interval) / maxCount;
    
    %% Nonlinear Features
    % Poincaré plot
    % Calculate SD1
    intervals_sec = interval / 1000; % Convert back to seconds for standard calculation
    diff_RR = diff(intervals_sec);
    PoincareSD1 = std(diff_RR) / sqrt(2);
    
    %% Update time-domain features structure
    timeDomain.meanNN = [timeDomain.meanNN, meanNN]; 
    timeDomain.medianNN = [timeDomain.medianNN, medianNN]; 
    timeDomain.SDNN = [timeDomain.SDNN, SDNN]; 
    timeDomain.RMSSD = [timeDomain.RMSSD, RMSSD];
    timeDomain.PNN50 = [timeDomain.PNN50, PNN50];
    timeDomain.meanRate = [timeDomain.meanRate, meanRate];
    timeDomain.sdRate = [timeDomain.sdRate, sdRate];
    timeDomain.HRVTi = [timeDomain.HRVTi, HRVTi];
    
    % Update non-linear features structure
    nonlinear.PoincareSD1 = [nonlinear.PoincareSD1, PoincareSD1];
end

end