function main()

    % The user is prompted to select an image 
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files'}, 'Select an image file');
    
    % Check if the user canceled the selection
    if isequal(filename,0) || isequal(pathname,0)
        disp('User canceled the operation.');
        return;
    end

    % Reading the selected image
    img = imread(fullfile(pathname, filename));

    % Display the image for selecting a pixel
    figure;
    imshow(img);
    title('Select a point on the object to extract');

    % Wait for the user to select a point
    [x, y] = ginput(1);
    x = round(x);
    y = round(y);

    % Get the RGB values of the selected pixel
    selectedPixelRGB = squeeze(double(img(y, x, :))) / 255;

    % Output the specific RGB values of the selected pixel, scaled to [0, 255] and rounded to 2 decimal places
    disp(['Selected pixel RGB: [', ...
        num2str(selectedPixelRGB(1) * 255, '%.2f'), ', ', ...
        num2str(selectedPixelRGB(2) * 255, '%.2f'), ', ', ...
        num2str(selectedPixelRGB(3) * 255, '%.2f'), ']', newline]);


    % Get the colour name of the selected pixel
    colourNameSelected = getColourName(selectedPixelRGB);
    disp(['Selected pixel colour name: ', colourNameSelected, newline]);

    Displaycolour(selectedPixelRGB, 'Selected Pixel Colour');

    % Extract region with the same colour using region growing
    % Adjust the threshold 
    detectedMask = RegionGrowing(img, selectedPixelRGB * 255, 80, x, y);

     % Display the extracted region with the same colour as the selected pixel
    figure;
    imshow(detectedMask);
    title('Region Growing Result');

    % Allow the user to draw a ground truth mask around the selected colour region
    groundTruthMask = createGroundTruthMask(img);

    % Display the ground truth region
    figure;
    imshow(groundTruthMask);
    title('Ground Truth Mask');

    % Create and display the comparison overlay
    comparisonOverlay = createComparisonOverlay(img, detectedMask, groundTruthMask);
    figure;
    imshow(comparisonOverlay);
    title('Comparison Overlay');

    % Calculate the average RGB values in the detected region
    averageRGB = CalculateAverageRGB(img, detectedMask);
    
    % Display the average RGB values rounded to 2 decimal places
    disp(['Average RGB in Region: [', ...
        num2str(averageRGB(1), '%.2f'), ', ', ...
        num2str(averageRGB(2), '%.2f'), ', ', ...
        num2str(averageRGB(3), '%.2f'), ']', newline]);

    Displaycolour(averageRGB, 'Average Region colour');

    % Evaluate the segmentation
    [TP, FP, TN, FN, accuracy, precision, recall] = evaluateSegmentation(groundTruthMask, detectedMask);
    
    % Display results
    disp(['True Positives: ', num2str(TP)]);
    disp(['False Positives: ', num2str(FP)]);
    disp(['True Negatives: ', num2str(TN)]);
    disp(['False Negatives: ', num2str(FN), newline, newline]);
    disp(['Accuracy: ', num2str(accuracy * 100, '%.2f'), '%']);
    disp(['Precision: ', num2str(precision * 100, '%.2f'), '%']);
    disp(['Sensitivity: ', num2str(recall * 100, '%.2f'), '%', newline, newline]);

    % Calculate discrepancies
    euclideanDist = colourDistance(selectedPixelRGB, averageRGB);
    
    % Display results
    disp(['Euclidean Distance: ', num2str(euclideanDist)]);

end

%-----------------------------------------------------------------------------------%

function region = RegionGrowing(img, seedcolour, thresholdRange, seedX, seedY)
    % Get image dimensions
    [rows, cols, ~] = size(img);

    % Convert seed colour to double
    seedcolour = double(seedcolour);

    % Initialize the region mask
    region = false(rows, cols);

    % Initialize queue for BFS
    queue = [seedX, seedY];

    % Perform BFS to find connected region
    while ~isempty(queue)
        % Pop the first element from the queue
        current = queue(1,:);
        queue(1,:) = [];

        % Extract coordinates
        cx = current(1);
        cy = current(2);

        % Check if the pixel is within the image boundaries
        if cx >= 1 && cx <= cols && cy >= 1 && cy <= rows
            % Check if the pixel is already in the region
            if ~region(cy, cx)
                % Get the colour of the current pixel
                currentcolour = double(squeeze(img(cy, cx, :)));

                % Calculate the Euclidean distance between the current pixel and the seed colour
                distance = sqrt(sum((currentcolour - seedcolour).^2));

                % Check if the distance is within the threshold range
                if distance <= thresholdRange
                    % Add pixel to the region
                    region(cy, cx) = true;

                    % Add neighbouring pixels to the queue
                    neighbours = [cx+1, cy; cx-1, cy; cx, cy+1; cx, cy-1];
                    validneighbours = neighbours(:,1) >= 1 & neighbours(:,1) <= cols & ...
                                      neighbours(:,2) >= 1 & neighbours(:,2) <= rows;
                    neighbours = neighbours(validneighbours, :);

                    % Enqueue valid neighbours
                    queue = [queue; neighbours];
                end
            end
        end
    end

    % Post-process the region to include nearby pixels with similar colours
    expandedRegion = imfill(region, 'holes'); % Fill holes in the region
    se = strel('disk', 5); % Increase the size of the structuring element for dilation
    expandedRegion = imdilate(expandedRegion, se); % Dilate the region to include nearby pixels
    region = expandedRegion;
end

%-----------------------------------------------------------------------------------%

function colourName = getColourName(rgb)
    % Define thresholds for classification
    thresholds = struct(...
        'red', 140, 'orange', 167, 'yellow', 130, ...
        'green', 120, 'cyan', 120, 'blue', 100, ...
        'purple', 130, 'pink', 50, 'brown', 90, ...
        'gray', 180, 'black', 50, 'white', 220, ...
        'lightGreen', 180, 'darkGreen', 50, ...
        'darkRed', 90, 'darkBlue', 90, 'darkPink', 130);

    % Extract RGB values
    red = rgb(1) * 255;
    green = rgb(2) * 255;
    blue = rgb(3) * 255;

    % Determine colour based on thresholds
    if red > thresholds.red && green < thresholds.green && blue < thresholds.blue
        colourName = 'Red';
    elseif red > thresholds.darkPink && green < thresholds.pink && blue > thresholds.pink
        colourName = 'Fushcia';
    elseif red > thresholds.orange && green > 50 && green < 150 && blue < 50
        colourName = 'Orange';
    elseif red > thresholds.yellow && green > thresholds.yellow && blue < thresholds.blue
        colourName = 'Yellow';
    elseif red < 200 && green > thresholds.cyan && blue > 150
        colourName = 'Blue';
    elseif red < thresholds.green && green > thresholds.green && blue < thresholds.cyan
        colourName = 'Green';
    elseif red > thresholds.brown && green > thresholds.brown && blue < thresholds.brown
        colourName = 'Brown';
    elseif red > thresholds.pink && green < thresholds.pink && blue > thresholds.pink
        colourName = 'Purple';
    elseif red > thresholds.purple && green < thresholds.purple && blue > thresholds.purple
        colourName = 'Pink';
    elseif red > thresholds.gray && green > thresholds.gray && blue > thresholds.gray
        colourName = 'Gray';
    elseif red < thresholds.black && green < thresholds.black && blue < thresholds.black
        colourName = 'Black';
    elseif red > thresholds.white && green > thresholds.white && blue > thresholds.white
        colourName = 'White';
    elseif green > thresholds.lightGreen && blue < thresholds.cyan && red < thresholds.green
        colourName = 'Light Green';
    elseif green < thresholds.darkGreen && blue < thresholds.cyan && red < thresholds.darkGreen
        colourName = 'Dark Green';
    elseif red < thresholds.darkRed && green < thresholds.green && blue < thresholds.blue
        colourName = 'Dark Red';
    elseif blue < thresholds.darkBlue && green < thresholds.cyan && red < thresholds.red
        colourName = 'Brown';
    else
        colourName = 'Unkown colour';
    end
end

%-----------------------------------------------------------------------------------%

function averageRGB = CalculateAverageRGB(img, mask)
    % Find indices where the mask is true
    [rows, cols] = find(mask);
    
    % Initialize sum variables for RGB channels
    sumR = 0; sumG = 0; sumB = 0; count = numel(rows);
    
    % Sum up RGB values
    for i = 1:count
        sumR = sumR + double(img(rows(i), cols(i), 1));
        sumG = sumG + double(img(rows(i), cols(i), 2));
        sumB = sumB + double(img(rows(i), cols(i), 3));
    end
    
    % Calculate averages
    if count > 0
        averageRGB = [sumR, sumG, sumB] / count;
    else
        averageRGB = [0, 0, 0]; % Default to black if no pixels are selected
    end

end

%-----------------------------------------------------------------------------------%

function groundTruthMask = createGroundTruthMask(img)
    % Display the image and let the user draw the ground truth region freehand
    figure;
    imshow(img);
    title('Draw the ground truth region freehand and double-click to finish');

    % Use the drawfreehand tool to allow the user to manually outline the ground truth region
    h = drawfreehand('Colour', 'green');

    % Create a binary mask from the drawn region
    groundTruthMask = createMask(h);
end


%-----------------------------------------------------------------------------------%

function [TP, FP, TN, FN, accuracy, precision, recall] = evaluateSegmentation(groundTruthMask, detectedMask)
    % Ensure both masks are logical and the same size
    groundTruthMask = logical(groundTruthMask);
    detectedMask = logical(detectedMask);
    
    % True Positives (TP)
    TP = sum(groundTruthMask(:) & detectedMask(:));
    
    % False Positives (FP)
    FP = sum(~groundTruthMask(:) & detectedMask(:));
    
    % True Negatives (TN)
    TN = sum(~groundTruthMask(:) & ~detectedMask(:));
    
    % False Negatives (FN)
    FN = sum(groundTruthMask(:) & ~detectedMask(:));
    
    % Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    % Precision
    precision = TP / (TP + FP);
    
    % Sensitivity
    recall = TP / (TP + FN);
end

%-----------------------------------------------------------------------------------%

function overlayImage = createComparisonOverlay(originalImage, detectedMask, groundTruthMask)
    % Ensure the image is in RGB
    if size(originalImage, 3) == 1
        overlayImage = cat(3, originalImage, originalImage, originalImage);
    else
        overlayImage = originalImage;
    end
    
    % Creating the colour masks using masks created from the previous
    % algorithms
    % The symbol ~ is a logical NOT operator
    truePositiveMask = detectedMask & groundTruthMask;   % Both masks agree
    falsePositiveMask = detectedMask & ~groundTruthMask; % Detected but not in ground truth
    falseNegativeMask = ~detectedMask & groundTruthMask; % In ground truth but not detected
   
    % Initialize a black image with the same dimensions as the original
    % image
    overlayImage = uint8(zeros(size(overlayImage))); 

    % Assign colours to different conditions
    overlayImage(:,:,1) = uint8(255 * falsePositiveMask); % Red for False Positives
    overlayImage(:,:,2) = uint8(255 * truePositiveMask);  % Green for True Positives
    overlayImage(:,:,3) = uint8(255 * falseNegativeMask); % Blue for False Negatives

    % Retain the original image in areas without labels
    noLabelMask = ~(truePositiveMask | falsePositiveMask | falseNegativeMask);
    for c = 1:3 % For each colour channel
        channel = overlayImage(:,:,c);
        origChannel = originalImage(:,:,c);
        channel(noLabelMask) = origChannel(noLabelMask);
        overlayImage(:,:,c) = channel;
    end
end

%-----------------------------------------------------------------------------------%

function Displaycolour(rgbcolour, titleText)
    % Ensure rgbcolour is within the correct range and type
    rgbcolour = double(rgbcolour);  % Ensure it's in double for manipulation
    if max(rgbcolour) <= 1
        rgbcolour = rgbcolour * 255;  % Assume the input was in [0, 1] and scale it
    end
    rgbcolour = uint8(rgbcolour);  % Convert to uint8 after scaling

    % Create a colour block
    colourBlock = repmat(reshape(rgbcolour, [1, 1, 3]), 100, 100);

    % Create a new figure for each colour display
    figure;
    imshow(colourBlock);
    title(titleText);
end

%-----------------------------------------------------------------------------------%

function dist = colourDistance(colour1, colour2)
    % Calculate the Euclidean distance between two RGB colours
    dist = sqrt(sum((colour1 - colour2).^2));
end

