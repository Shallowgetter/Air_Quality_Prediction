%你需要预测什么数据，就改文件名称，这里我用的是PM2.5的数据
% 从训练集的 CSV 文件中读取数据
train_data = readtable('matched_data_train_PM25.csv');
% 从测试集的 CSV 文件中读取数据
test_data = readtable('matched_data_test_PM25.csv');

% 特征工程：训练数据
date_str_train = num2str(train_data{:, 1}); % 第一列是日期
time_str_train = arrayfun(@(x) sprintf('%02d', x), train_data{:, 2}, 'UniformOutput', false); % 统一时间格式为两位数字
datetime_str_train = strcat(date_str_train, time_str_train);
datetime_train = datetime(datetime_str_train, 'InputFormat', 'yyyyMMddHH');
% 修改此处以仅包含日期和小时
datetime_train = dateshift(datetime_train, 'start', 'hour'); % 只保留小时，忽略日期

%特征工程：测试数据
date_str_test = num2str(test_data{:, 1}); % 第一列是日期
time_str_test = arrayfun(@(x) sprintf('%02d', x), test_data{:, 2}, 'UniformOutput', false); % 统一时间格式为两位数字
datetime_str_test = strcat(date_str_test, time_str_test);
datetime_test = datetime(datetime_str_test, 'InputFormat', 'yyyyMMddHH');
% 修改此处以仅包含日期和小时
datetime_test = dateshift(datetime_test, 'start', 'hour'); % 只保留小时，忽略日期


%需要哪个城区的数据，就修改对应的数字，这里用的是东城东四的数据，所以train_data和test_data这里是4
% 提取特征和目标变量
X_train = [day(datetime_train), hour(datetime_train)]; % 使用日和小时作为特征
%train_data(:, 4)表示东城东四的PM2.5的数据，
% 5表示：东城天坛的数据，
% 6：西城官园
% 7：西城万寿西宫
% 8：朝阳奥体中心
% 9：朝阳农展馆
% 10：海淀万柳
% 11：海淀四季青
% 12：丰台小屯
% 13：丰台云岗
% 14：石景山古城
y_train = train_data{:, 4}; 

X_test = [day(datetime_test), hour(datetime_test)]; % 使用日和小时作为特征
%test_data(:, 4)表示东城东四的PM2.5的数据，
% 5表示：东城天坛的数据，
% 6：西城官园
% 7：西城万寿西宫
% 8：朝阳奥体中心
% 9：朝阳农展馆
% 10：海淀万柳
% 11：海淀四季青
% 12：丰台小屯
% 13：丰台云岗
% 14：石景山古城
y_test = test_data{:, 4}; 

% 构建随机森林模型
ntrees = 100; % 设置决策树的数量
min_leaf_size = 7; % 设置叶子节点的最小样本数为10
mdl = TreeBagger(ntrees, X_train, y_train, 'MinLeafSize', min_leaf_size);

% 在测试集上进行预测
predicted_values = str2double(predict(mdl, X_test));

% 评估模型
mse = mean((predicted_values - y_test).^2); % 计算均方误差

% 显示结果
disp(['使用叶子节点最小样本数为10时的均方误差（MSE）: ', num2str(mse)]);

% 计算每个特征与预测误差之间的关系
% 对日进行方差分析
[p_day, tbl_day, stats_day] = anova1(predicted_values - y_test, X_test(:, 1));

% 对小时进行方差分析
[p_hour, tbl_hour, stats_hour] = anova1(predicted_values - y_test, X_test(:, 2));

% 显示结果
disp(['日对预测误差的方差分析结果: p-value = ', num2str(p_day)]);
disp(['小时对预测误差的方差分析结果: p-value = ', num2str(p_hour)]);


