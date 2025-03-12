%训练数据
% 文件夹路径
folder_path_train = 'data';
% 获取文件夹中所有CSV文件
files_train = dir(fullfile(folder_path_train, '*.csv'));
% 创建一个空的表格来存储所有匹配的数据
matched_data_train = table();

% 循环读取每个文件
for i = 1:length(files_train)
    % 生成完整文件路径
    file_path_train = fullfile(folder_path_train, files_train(i).name);
    % 读取CSV文件
    data_train = readtable(file_path_train);
    % 获取第三列的列名
    column_names_train = data_train.Properties.VariableNames;
    third_column_name_train = column_names_train{3};
    % 要匹配的名称是 'PM2.5'
    desired_name_train = 'PM2.5';
    % 查找匹配的列索引
    column_index_train = find(strcmp(column_names_train, third_column_name_train));
    % 找到匹配的行索引
    matching_rows_train = strcmp(data_train{:, column_index_train}, desired_name_train);
    % 提取匹配的数据
    matched_data = data_train(matching_rows_train, :);
    % 将匹配的数据添加到匹配的数据集中
    matched_data_train = [matched_data_train; matched_data];
end
% 检查并替换训练集中的空值
for i = 1:size(matched_data_train, 2) % 遍历每一列
    missing_indices = ismissing(matched_data_train{:, i}); % 找到空值的索引
    if any(missing_indices) % 如果存在空值
        % 找到第一个非空值的索引
        first_non_missing_index = find(~missing_indices, 1, 'first');
        % 使用上一个非空值来替换空值
        matched_data_train{missing_indices, i} = matched_data_train{first_non_missing_index, i};
    end
end

% 保存所有匹配的数据到一个CSV文件中
output_file_train = 'matched_data_train_PM25.csv';
writetable(matched_data_train, output_file_train);



%测试数据
% 文件夹路径
folder_path_test = 'test';
% 获取文件夹中所有CSV文件
files_test = dir(fullfile(folder_path_test, '*.csv'));
% 创建一个空的表格来存储所有匹配的数据
matched_data_test = table();

% 循环读取每个文件
for i = 1:length(files_test)
    % 生成完整文件路径
    file_path_test = fullfile(folder_path_test, files_test(i).name);
    % 读取CSV文件
    data_test = readtable(file_path_test);
    % 获取第三列的列名
    column_names_test = data_test.Properties.VariableNames;
    third_column_name_test = column_names_test{3};
    % 要匹配的名称是 'PM2.5'
    desired_name_test = 'PM2.5';
    % 查找匹配的列索引
    column_index_test = find(strcmp(column_names_test, third_column_name_test));
    % 找到匹配的行索引
    matching_rows_test = strcmp(data_test{:, column_index_test}, desired_name_test);
    % 提取匹配的数据
    matched_data = data_test(matching_rows_test, :);
    % 将匹配的数据添加到匹配的数据集中
    matched_data_test = [matched_data_test; matched_data];
end
% 检查并替换测试集中的空值
for i = 1:size(matched_data_test, 2) % 遍历每一列
    missing_indices = ismissing(matched_data_test{:, i}); % 找到空值的索引
    if any(missing_indices) % 如果存在空值
        % 找到第一个非空值的索引
        first_non_missing_index = find(~missing_indices, 1, 'first');
        % 使用上一个非空值来替换空值
        matched_data_test{missing_indices, i} = matched_data_test{first_non_missing_index, i};
    end
end
% 保存所有匹配的数据到一个CSV文件中
output_file_test = 'matched_data_test_PM25.csv';
writetable(matched_data_test, output_file_test);


