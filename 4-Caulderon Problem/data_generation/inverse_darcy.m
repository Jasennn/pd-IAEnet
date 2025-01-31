%% Settings

num_data = 5;
split_size = 250;

media.phantom_size = 61;
media.phantom_type = 'msl';
media.padding = 1;

media.size = media.phantom_size + 2 * media.padding; % real media size is this + 1.

boundary.size = 4*media.size;

meas.size = boundary.size;

generate_dataset(num_data, split_size, media, meas, boundary)
% dm = pairwiseDistanceMatrix(data_meas(:, 1:2:end, 1:2:end))

% sample_media = generate_shepp_logan(media);
% imagesc(sample_media)
% colorbar

% tic
% [coeff, meas_matrix] = generate_sample(media, meas, boundary);
% toc
% 
% tic
% [coeff2, meas_matrix2] = generate_sample(media, meas, boundary);
% toc
% 
% norm(meas_matrix - meas_matrix2)
% norm(meas_matrix(1:2:end, 1:2:end) - meas_matrix2(1:2:end, 1:2:end))
% norm(meas_matrix(1:3:end, 1:3:end) - meas_matrix2(1:3:end, 1:3:end))
% 
% figure
% imagesc(coeff)
% colorbar;
% 
% figure
% imagesc(meas_matrix)
% colorbar
% 
% figure
% imagesc(coeff2)
% colorbar;
% 
% figure
% imagesc(meas_matrix2)
% colorbar

%% Sample Diversity
function distanceMatrix = pairwiseDistanceMatrix(dataset)
    % Calculate the pairwise distance matrix for a dataset of matrices.
    %
    % Args:
    % - dataset: A 3D array of size n x m x m, where n is the number of matrices, and each matrix is of size m x m.
    %
    % Returns:
    % - distanceMatrix: An n x n distance matrix.

    n = size(dataset, 1);
    distanceMatrix = zeros(n, n);

    for i = 1:n
        for j = 1:n
            distanceMatrix(i, j) = norm(dataset(i, :, :) - dataset(j, :, :), 'fro');
        end
    end
end

%% Generate dataset
function generate_dataset(num_data, split_size, media, meas, boundary)
    num_splits = ceil(num_data / split_size);
    
    data_media = zeros(num_data, media.size + 1, media.size + 1);
    data_meas = zeros(num_data, meas.size, meas.size);

    for i = 1:num_splits
        fprintf("split %d/%d\n", i, num_splits)

        start_idx = (i-1)*split_size + 1;
        end_idx = min(i*split_size, num_data);
        curr_split_size = end_idx - start_idx + 1;

        split_media = zeros(curr_split_size, media.size + 1, media.size + 1);
        split_meas = zeros(curr_split_size, meas.size, meas.size);
        parfor j = 1:curr_split_size
            disp(j)
            tic
            [coeff, meas_matrix] = generate_sample(media, meas, boundary);
            toc
            split_media(j, :, :) = coeff;
            split_meas(j, :, :) = meas_matrix;
        end
        data_media(start_idx:end_idx, :, :) = split_media;
        data_meas(start_idx:end_idx, :, :) = split_meas;
        save(sprintf('inv_darcy_shepp_logan%d.mat', num_data), ...
            'data_media', 'data_meas', '-v7.3');
    end 
end

%% Generate media
function media_data = generate_shepp_logan(media)
    media_data = random_Shepp_Logan(media.phantom_size, ...
        {'pad', media.padding; 'M', 1; 'phantom', media.phantom_type});
    media_data = reshape(media_data, media.size, media.size);
    media_data = [media_data, zeros(media.size, 1); zeros(1, media.size + 1)];
    a = min(min(media_data));
    b = max(max(media_data));
    media_data = 10*(media_data - a) / (b - a) + 1;
    media_data = exp(media_data);
end

%% Generate measurement matrix
function meas_data = generate_meas(media_data, media, meas, boundary)
    meas_data = zeros(meas.size, meas.size);
    for j = 1:boundary.size
%         forcing_data = zeros(media.size, media.size);
        d = zeros(boundary.size, 1);
        d(j) = 1;
        [meas_row, sol] = EllipticSolver(d, media.size, media_data);
        meas_data(j, :) = meas_row;
        
%         if j == 30
%             figure
%             imagesc(sol)
%             colorbar
%         end
    end
end

%% Generate sample
function [media_data, meas_data] = generate_sample(media, meas, boundary)
    media_data = generate_shepp_logan(media);
    meas_data = generate_meas(media_data, media, meas, boundary);
end