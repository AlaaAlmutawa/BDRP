sourceDir = 'blurwine/images_1/';  % Specify source directory
destDir = 'deblurwine'; % Specify destination directory

% % Create the destination directory if it doesn't exist
% if ~exist(destDir, 'dir')
%     mkdir(destDir);
% end

disp(['Current directory: ', pwd]);
disp(['directories: ', ls]);
if exist(sourceDir, 'dir')
    disp(['Directory found: ', sourceDir]);
else
    disp(['Directory not found: ', sourceDir]);
end
% List all jpg files in the source directory
imageFiles = dir(fullfile(sourceDir, '*.png'));  
len = length(imageFiles);
disp(['Length of the array: ', num2str(len)]);
for file = imageFiles'
    % Read each image
    imagePath = fullfile(file.folder, file.name);
    Image = im2double(imread(imagePath));
    
    % Deblurring logic
    % t1 = gaussian, [3,3], 2 default itr w sharpening
    % t2 = gaussian, [5,5], 2 default itr w sharpening
    % t3 = gaussian, [5,5], 2 default itr w/o sharpening
    % t4 = gaussian, [5,5], 5 default itr w/o sharpening
    % t5 = gaussian, [5,5], 2 itr 30 w/o sharpening
    % t6 = gaussian, [5,5], 2 itr 15 w/ sharpening
    initialPSF = fspecial('gaussian', [5, 5], 2);
    [height, width, ~] = size(Image);
    deblurredImage = zeros(height, width, 3); % Preallocate memory

    for i = 1:3
        [deblurredChannel, ~] = deconvblind(Image(:,:,i), initialPSF,15);
        deblurredImage(:,:,i) = deblurredChannel;
    end

    deblurredImage = imsharpen(deblurredImage);
    
    % Save the deblurred image
    [~, imageName, ext] = fileparts(file.name);
    imwrite(deblurredImage, fullfile(destDir, [imageName, ext]));
end