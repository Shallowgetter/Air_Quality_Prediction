import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 数据读取和预处理函数
def read_and_preprocess_data(directory, selected_sites):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for site in selected_sites:
                    site_data = data.get(site)
                    if site_data:
                        for hour, hour_data in site_data.items():
                            pm25 = hour_data.get('PM2.5', np.nan)
                            pm10 = hour_data.get('PM10', np.nan)
                            aqi = hour_data.get('AQI', np.nan)
                            date_str = filename.split('_')[-1].split('.')[0]
                            time_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {hour}:00:00"
                            record = {
                                'site': site,
                                'time': time_str,
                                'PM2.5': pm25,
                                'PM10': pm10,
                                'AQI': aqi
                            }
                            all_data.append(record)
    df = pd.DataFrame(all_data)
    imputer = SimpleImputer(strategy='mean')
    df[['PM2.5', 'PM10', 'AQI']] = imputer.fit_transform(df[['PM2.5', 'PM10', 'AQI']])
    return df

def read_test_data(directory, selected_sites):
    test_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for site in selected_sites:
                    site_data = data.get(site)
                    if site_data:
                        for hour, hour_data in site_data.items():
                            date_str = filename.split('_')[-1].split('.')[0]
                            time_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {hour}:00:00"
                            record = {
                                'site': site,
                                'time': time_str,
                                'PM2.5': hour_data.get('PM2.5', np.nan),
                                'PM10': hour_data.get('PM10', np.nan),
                                'AQI': hour_data.get('AQI', np.nan)
                            }
                            test_data.append(record)
    df_test = pd.DataFrame(test_data)
    return df_test

# 定义数据集
class AirQualityDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_train_data():
    selected_sites = [
        '东城东四', '东城天坛', '西城官园', '西城万寿西宫', '朝阳奥体中心',
        '朝阳农展馆', '海淀万柳', '海淀四季青', '丰台小屯', '丰台云岗', '石景山古城'
    ]
    data_directory = r'D:\统计建模2024\北京（18年到23年）\111_json'
    df = read_and_preprocess_data(data_directory, selected_sites)
    scaler = StandardScaler()
    features = df[['PM2.5', 'PM10', 'AQI']].values
    targets = df[['PM2.5', 'PM10', 'AQI']].values
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    train_dataset = AirQualityDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = AirQualityDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, scaler, selected_sites

def run_test_predictions(model, scaler, selected_sites):
    test_directory = r'D:\统计建模2024\北京（18年到23年）\111_test'
    df_test = read_test_data(test_directory, selected_sites)
    features_test = scaler.transform(df_test[['PM2.5', 'PM10', 'AQI']].values)
    test_dataset = AirQualityDataset(torch.tensor(features_test, dtype=torch.float32), torch.zeros(len(features_test), 3))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_predictions = model(batch_features)
            predictions.extend(batch_predictions.cpu().numpy())
    df_test['PM2.5_predicted'] = [pred[0] for pred in predictions]
    df_test['PM10_predicted'] = [pred[1] for pred in predictions]
    df_test['AQI_predicted'] = [pred[2] for pred in predictions]
    for col, pred_col in [('PM2.5', 'PM2.5_predicted'), ('PM10','PM10_predicted'), ('AQI', 'AQI_predicted')]:
         df_test[col] = df_test[col].fillna(df_test[col].mean())
         df_test[pred_col] = df_test[pred_col].fillna(df_test[pred_col].mean())
    from sklearn.metrics import mean_squared_error
    mse_pm25 = mean_squared_error(df_test['PM2.5'], df_test['PM2.5_predicted'])
    mse_pm10 = mean_squared_error(df_test['PM10'], df_test['PM10_predicted'])
    mse_aqi = mean_squared_error(df_test['AQI'], df_test['AQI_predicted'])
    print(f'MSE for PM2.5: {mse_pm25}')
    print(f'MSE for PM10: {mse_pm10}')
    print(f'MSE for AQI: {mse_aqi}')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for site in selected_sites:
         site_data = df_test[df_test['site'] == site].sort_values(by='time')
         site_data['hour'] = pd.to_datetime(site_data['time']).dt.hour
         plt.figure(figsize=(15, 5))
         plt.scatter(site_data['hour'], site_data['PM2.5'], label='PM2.5 Real', color='blue', marker='o')
         plt.scatter(site_data['hour'], site_data['PM10'], label='PM10 Real', color='green', marker='o')
         plt.scatter(site_data['hour'], site_data['AQI'], label='AQI Real', color='red', marker='o')
         plt.scatter(site_data['hour'], site_data['PM2.5_predicted'], label='PM2.5 Predicted', color='skyblue', marker='x')
         plt.scatter(site_data['hour'], site_data['PM10_predicted'], label='PM10 Predicted', color='lightgreen', marker='x')
         plt.scatter(site_data['hour'], site_data['AQI_predicted'], label='AQI Predicted', color='salmon', marker='x')
         plt.title(f'Air Quality Predictions for {site}')
         plt.xlabel('Hour of the Day')
         plt.ylabel('Value')
         plt.legend()
         plt.grid(True)
         plt.show()
    return df_test

def save_preprocessed_data():
    selected_sites = [
        '东城东四', '东城天坛', '西城官园', '西城万寿西宫', '朝阳奥体中心',
        '朝阳农展馆', '海淀万柳', '海淀四季青', '丰台小屯', '丰台云岗', '石景山古城'
    ]
    data_directory = r'D:\统计建模2024\北京（18年到23年）\111_json'
    df = read_and_preprocess_data(data_directory, selected_sites)
    os.makedirs(r'D:\Air_Quality_Prediction\data', exist_ok=True)
    output_path = r'D:\Air_Quality_Prediction\data\on_process\air_quality_data.csv'
    df.to_csv(output_path, index=False)
    print(f'中间数据文件已保存到: {output_path}')
