function [timeDomain, nonlinear] = ...
    HRVfeature_mmwave(d2f, winSize, winStride, b1, b2, m, max_iter, fs)
% --- Initialization ---
% Initialize empty structures with array fields
timeDomain = struct('meanNN', [], 'medianNN', [], 'SDNN', [], ...
                    'RMSSD', [], 'PNN50', [], ...
                    'meanRate', [], 'sdRate', [], 'HRVTi', []);
nonlinear = struct('PoincareSD1', []);

% Calculate the number of windows
numWindows = floor((length(d2f) - winSize) / winStride) + 1;

% Define convergence threshold
tolerance = 1e-5; 

% Initialize heartbeat template for warm start
miu_template = zeros(m, 1); 

% --- Sliding Window Processing ---
for i = 1:numWindows
    % Get the data for the current window
    startIdx = (i-1) * winStride + 1;
    endIdx = startIdx + winSize - 1;
    current_d2f = d2f(startIdx : endIdx);
   
    % Call the heartbeat segmentation function
    [S, miu_template] = heartbeatSeg(current_d2f, b1, b2, m, max_iter, miu_template, tolerance);
    
    % --- Feature Calculation ---
    if isempty(S) || any(cellfun(@isempty, S))
        % If segmentation fails, fill with NaN and continue to the next window
        timeDomain.meanNN = [timeDomain.meanNN, NaN]; 
        timeDomain.medianNN = [timeDomain.medianNN, NaN]; 
        timeDomain.SDNN = [timeDomain.SDNN, NaN]; 
        timeDomain.RMSSD = [timeDomain.RMSSD, NaN];
        timeDomain.PNN50 = [timeDomain.PNN50, NaN];
        timeDomain.meanRate = [timeDomain.meanRate, NaN];
        timeDomain.sdRate = [timeDomain.sdRate, NaN];
        timeDomain.HRVTi = [timeDomain.HRVTi, NaN];
        nonlinear.PoincareSD1 = [nonlinear.PoincareSD1, NaN];
        continue;
    end
    
    interval = cellfun(@length, S);
    interval = interval ./ fs;      % unit: seconds
    interval = interval .* 1000;    % unit: ms
    
    %% Time Domain
    meanNN = mean(interval);
    medianNN = median(interval);
    SDNN = std(interval);
    
    if numel(interval) >= 2
        diffs = diff(interval);
        RMSSD = sqrt(mean(diffs.^2));
        PNN50 = sum(abs(diffs) > 50) / numel(diffs);
    else
        RMSSD = NaN; 
        PNN50 = NaN;
    end
    
    rate = (60*1000) ./ interval; % unit: bpm
    meanRate = mean(rate);
    sdRate = std(rate);
    
    binWidth = 7.8125;
    [counts, ~] = histcounts(interval, 'BinWidth', binWidth);
    maxCount = max(counts);
    HRVTi = numel(interval) / maxCount;
    
    %% Nonlinear
    intervals_sec = interval / 1000;
    diff_RR = diff(intervals_sec);
    PoincareSD1 = std(diff_RR) / sqrt(2);
    
    %% Update feature structures
    timeDomain.meanNN = [timeDomain.meanNN, meanNN]; 
    timeDomain.medianNN = [timeDomain.medianNN, medianNN]; 
    timeDomain.SDNN = [timeDomain.SDNN, SDNN]; 
    timeDomain.RMSSD = [timeDomain.RMSSD, RMSSD];
    timeDomain.PNN50 = [timeDomain.PNN50, PNN50];
    timeDomain.meanRate = [timeDomain.meanRate, meanRate];
    timeDomain.sdRate = [timeDomain.sdRate, sdRate];
    timeDomain.HRVTi = [timeDomain.HRVTi, HRVTi];
    nonlinear.PoincareSD1 = [nonlinear.PoincareSD1, PoincareSD1];
end
end

%% Heartbeat Segmentation Method Based on Dynamic Programming
function [S, miu] = heartbeatSeg(d2f, b1, b2, m, max_iter, miu_init, tolerance)
x = d2f;
miu = miu_init; % Use the provided initial template

% Iterative update process
for iter = 1 : max_iter
    miu_old = miu;
    % Call the segmentation function
    S = optimal_segmentation(x, miu_old, b1, b2);
    
    % If segmentation fails (e.g., signal is too short or problematic), return immediately
    if isempty(S)
        miu = miu_old; % Return the old template
        return;
    end
    
    miu = update_template(S, m);
    
    % Check for convergence
    % Calculate the relative change between the new and old templates, adding eps to prevent division by zero
    change = norm(miu - miu_old) / (norm(miu_old) + eps);
    if change < tolerance
        break; % If the change is small enough, exit the loop early
    end
end
end

%% Helper Function 1: Optimal Segmentation
function S = optimal_segmentation(x, miu, b1, b2)
    n = length(x);
    L = length(miu);
    
    % --- Pre-compute interpolated templates ---
    % Use containers.Map for storage, with segment length M as the key and the interpolated template as the value
    precomputed_w = containers.Map('KeyType', 'double', 'ValueType', 'any');
    original_x_interp = 1:L;
    miu_col = miu(:);
    for M = b1:b2
        if M > 0
            query_x_interp = linspace(1, L, M);
            w_miu = interp1(original_x_interp, miu_col, query_x_interp, 'spline');
            precomputed_w(M) = w_miu(:);
        end
    end
    % --------------------------
    
    D = inf(n + 1, 1);
    D(1) = 0;
    prev = zeros(n + 1, 1);
    x_col = x(:);
    
    for t = 1:n
        min_tau = max(0, t - b2);
        max_tau = t - b1;
        
        if max_tau < min_tau
            continue;
        end
        
        for tau = min_tau:max_tau
            % tau cannot be equal to t, because the segment length M = t-tau must be greater than 0
            if tau == t
                continue;
            end
            
            % Calculate segment length and get the interpolated template from the pre-computed map
            M = t - tau;
            
            % Check if M is within the pre-computed range
            if ~isKey(precomputed_w, M)
                continue;
            end
            
            w_miu = precomputed_w(M);
            
            x_segment = x_col(tau + 1 : t);
            
            current_cost = sum((x_segment - w_miu).^2);
            if D(tau + 1) + current_cost < D(t + 1)
                D(t + 1) = D(tau + 1) + current_cost;
                prev(t + 1) = tau;
            end
        end
    end
    
    % Backtrack to get the segmentation results
    S = {};
    current_pos = n;
    while current_pos > 0
        prev_tau = prev(current_pos + 1);
        segment = x_col(prev_tau + 1 : current_pos);
        S = [{segment}, S];
        current_pos = prev_tau;
        if prev_tau == 0 % Reached the beginning
            break;
        end
    end
end

%% Helper Function 2: Template Update
function new_miu = update_template(segments, m)
    if isempty(segments) || any(cellfun(@isempty, segments))
        new_miu = zeros(m, 1); % If there are no valid segments, return a zero template
        return;
    end
    
    n = sum(cellfun(@length, segments));
    new_miu = zeros(m, 1);
    
    for i = 1:length(segments)
        si = segments{i};
        L = length(si);
        
        if L == 0; continue; end
        
        original_points = 1:L;
        query_points = linspace(1, L, m);
        w_si = interp1(original_points, si(:), query_points, 'spline');
        new_miu = new_miu + L * w_si(:);
    end
    
    if n > 0
        new_miu = new_miu / n;
    else
        % If the total length is 0, avoid division by zero error
        new_miu = zeros(m, 1);
    end
end