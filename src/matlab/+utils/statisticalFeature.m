function [means, sd, meanAbsDiff, meanAbsDiffNor, meanAbsDiff2, meanAbsDiff2Nor] ...
    = statisticalFeature(data, winSize, winStride)

% Get the length of the data
N = length(data);

% Calculate the number of windows (using floor to ensure it doesn't exceed data range)
numWindows = floor((N - winSize) / winStride) + 1;

% Pre-allocate output variables (for performance)
means = zeros(1, numWindows);
sd = zeros(1, numWindows);
meanAbsDiff = zeros(1, numWindows);
meanAbsDiffNor = zeros(1, numWindows);
meanAbsDiff2 = zeros(1, numWindows);
meanAbsDiff2Nor = zeros(1, numWindows);

% Main loop for sliding window processing
for i = 1 : numWindows
    % Calculate the start and end indices of the current window
    startIndex = (i-1) * winStride + 1;
    endIndex = startIndex + winSize - 1;
    
    % Extract window data
    windowData = data(startIndex : endIndex);
    
    % Calculate basic statistics
    means(i) = mean(windowData);  % Mean of the window
    sd(i) = std(windowData);      % Standard deviation of the window
    
    % ===== First-order difference features =====
    % Calculate the absolute value of the difference between adjacent elements
    diffData = abs(diff(windowData));
    
    % Calculate the mean of the absolute differences
    meanAbsDiff(i) = mean(diffData);
    
    % Normalized mean of the first-order difference
    meanAbsDiffNor(i) = meanAbsDiff(i) / sd(i);
    
    % ===== Second-order difference features =====
    % Check if the second-order difference can be calculated (requires at least 3 data points)
    if length(diffData) > 1
        % Calculate the absolute value of the second-order difference (difference of the first-order difference)
        diffData2 = abs(diff(diffData));
        meanAbsDiff2(i) = mean(diffData2);
    else
        % If the window is too small to calculate the second-order difference, set to NaN
        meanAbsDiff2(i) = NaN; 
    end
    
    % Normalized mean of the second-order difference
    meanAbsDiff2Nor(i) = meanAbsDiff2(i) / sd(i);
end
end