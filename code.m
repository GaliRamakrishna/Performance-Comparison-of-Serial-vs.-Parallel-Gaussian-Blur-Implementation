function main()
    % Specify the image path
    imagePath = "C:\Users\galir\Downloads\final project\Gaussian_Blur.jpg";

    % Read the image
    input_image = imread(imagePath);

    % Convert to grayscale if necessary (optional)
    if size(input_image, 3) == 3
        input_image = rgb2gray(input_image);
    end

    % Set the sigma value for Gaussian blur
    sigma = 1;

    % Define number of repetitions for timing
    num_repeats = 10;

    % Number of cores to test
    core_counts = [2, 4, 8]; % Adjust based on your system's capabilities

    % Initialize arrays to store execution times and results
    serial_times = zeros(num_repeats, 1);
    parallel_times = zeros(num_repeats, length(core_counts));

    % Measure execution time for serial implementation using manual Gaussian blur
    for i = 1:num_repeats
        tic;
        output_image_serial = gaussian_blur_manual(input_image, sigma);
        serial_times(i) = toc;
    end

    % Measure execution time for parallel implementations with different core counts
    for c = 1:length(core_counts)
        num_workers = core_counts(c);
        
        % Check if a parallel pool already exists; if not, create one
        pool = gcp('nocreate'); % Get current parallel pool (if it exists)
        if isempty(pool)
            parpool(num_workers); % Start a parallel pool with specified number of workers
        end

        for i = 1:num_repeats
            tic;
            output_image_parallel = gaussian_blur_parallel_manual(input_image, sigma, num_workers);
            parallel_times(i, c) = toc;
        end
        
        % Optionally delete the pool after use (uncomment if desired)
        % delete(gcp('nocreate')); 
    end

    % Calculate average times
    avg_serial_time = mean(serial_times);
    avg_parallel_times = mean(parallel_times);

    % Calculate speedup and efficiency for each core count
    speedup = avg_serial_time ./ avg_parallel_times;
    efficiency = speedup ./ core_counts'; % Divide by number of workers

    % Display results
    fprintf('Average Serial Time: %.4f seconds\n', avg_serial_time);
    
    for c = 1:length(core_counts)
        fprintf('Average Parallel Time (%d cores): %.4f seconds\n', core_counts(c), avg_parallel_times(c));
        fprintf('Speedup (%d cores): %.2f\n', core_counts(c), speedup(c));
        fprintf('Efficiency (%d cores): %.2f\n', core_counts(c), efficiency(c));
    end

    % Visualization of results
    figure;

    % Combine average times into a single array for plotting
    combined_avg_times = [avg_serial_time; avg_parallel_times(:)];

    % Execution Time Comparison Plot
    subplot(3, 1, 1);
    bar(combined_avg_times); 
    set(gca, 'XTickLabel', [{'Serial'}, arrayfun(@(x) sprintf('%d Cores', x), core_counts, 'UniformOutput', false)]);
    ylabel('Execution Time (seconds)');
    title('Execution Time Comparison');

    % Speedup Plot
    subplot(3, 1, 2);
    bar(speedup);
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d Cores', x), core_counts, 'UniformOutput', false));
    ylabel('Speedup');
    title('Speedup Comparison');

    % Efficiency Plot
    subplot(3, 1, 3);
    bar(efficiency);
    set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%d Cores', x), core_counts, 'UniformOutput', false));
    ylabel('Efficiency');
    title('Efficiency Comparison');
end

function output_image = gaussian_blur_manual(input_image, sigma)
   kernel_size = 2 * ceil(3 * sigma) + 1; % Calculate size of Gaussian kernel
   kernel_half_size = floor(kernel_size / 2);
   [X, Y] = meshgrid(-kernel_half_size:kernel_half_size, -kernel_half_size:kernel_half_size);
   
   % Create Gaussian kernel
   kernel = exp(-(X.^2 + Y.^2) / (2 * sigma^2));
   kernel = kernel / sum(kernel(:)); % Normalize the kernel
   
   [rows, cols] = size(input_image);
   output_image = zeros(rows, cols);

   % Apply convolution using the Gaussian kernel manually
   for i = 1:rows
       for j = 1:cols
           for kx = -kernel_half_size:kernel_half_size
               for ky = -kernel_half_size:kernel_half_size
                   if (i + kx > 0 && i + kx <= rows && j + ky > 0 && j + ky <= cols)
                       output_image(i,j) = output_image(i,j) + input_image(i + kx,j + ky) * kernel(kx + kernel_half_size + 1, ky + kernel_half_size + 1);
                   end
               end
           end
       end
   end

   output_image = uint8(output_image); % Convert back to uint8 if needed
end

function output_image = gaussian_blur_parallel_manual(input_image, sigma, num_workers)
   [rows, cols] = size(input_image);
   output_image = zeros(rows, cols);

   parfor i = 1:rows
       output_image(i,:) = gaussian_blur_manual(input_image(i,:), sigma); 
   end

   output_image = uint8(output_image); % Convert back to uint8 if needed
end

