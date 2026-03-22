import os
import pandas as pd

# 1. 扫描当前文件夹中所有CSV文件（包括子文件夹）
csv_files = []
for root, dirs, files in os.walk('.'):  # 扫描当前目录
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            csv_files.append(full_path)

# 2. 列出所有找到的CSV文件，让你选
print("🔍 找到以下CSV文件：")
for i, path in enumerate(csv_files):
    print(f"   [{i}] {path}")

if len(csv_files) == 0:
    print("❌ 没找到任何CSV文件！请检查当前文件夹中是否有CSV文件")
    exit()  # 退出程序，因为没有文件可以处理

# 3. 自动选第一个CSV文件（也可以手动改数字选其他）
selected_file = csv_files[0]
print(f"\n✅ 自动选择文件：{selected_file}")

# 4. 读取数据（终于能成功了！）
df = pd.read_csv(selected_file)
print(f"✅ 数据读取成功！")
print(f"数据形状：{df.shape} (行数, 列数)")
print(f"数据列名：{df.columns.tolist()}")

# ==================================================
# 下面是原来的预处理+建模代码（不用改）
# ==================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values('date').reset_index(drop=True)

# 构建预测目标：次日收盘价
df['next_day_close'] = df['close'].shift(-1)

# 构建时间序列特征
def create_time_features(df, window=5):
    for i in range(1, window+1):
        df[f'close_lag_{i}'] = df['close'].shift(i)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    return df

df = create_time_features(df)
df = df.dropna()
print(f"✅ 预处理完成！有效数据行数：{len(df)}")

# 划分训练/测试集
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# 定义特征
feature_cols = [col for col in df.columns if col.startswith('close_lag_') or col.startswith('SMA_')]
feature_cols += ['RSI_14', 'MACD', 'volatility_20'] if 'RSI_14' in df.columns else []
feature_cols = list(set(feature_cols) & set(df.columns))  # 只保留存在的列

X_train, y_train = train_data[feature_cols], train_data['next_day_close']
X_test, y_test = test_data[feature_cols], test_data['next_day_close']

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 计算指标
def calculate_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if (mask := y_true != 0).any() else 0
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name} 评估指标：")
    print(f"   MAE：{mae:.4f} 美元")
    print(f"   RMSE：{rmse:.4f} 美元")
    print(f"   MAPE：{mape:.2f}%")
    print(f"   R²：{r2:.4f}")
    return mae, rmse, mape, r2

train_mae, train_rmse, train_mape, train_r2 = calculate_metrics(y_train, y_pred_train, "训练集")
test_mae, test_rmse, test_mape, test_r2 = calculate_metrics(y_test, y_pred_test, "测试集")

# plot gragh
plt.figure(figsize=(14,7))
# Plot training set
plt.plot(train_data['date']+pd.Timedelta(days=1), y_train,
         color='#1f77b4', linewidth=1.5, label='Actual Close Price')
plt.plot(train_data['date']+pd.Timedelta(days=1), y_pred_train,
         color='#ff7f0e', linestyle='--', linewidth=1, label='Training Prediction')
# Plot test set
plt.plot(test_data['date']+pd.Timedelta(days=1), y_test,
         color='#1f77b4', linewidth=1.5)
plt.plot(test_data['date']+pd.Timedelta(days=1), y_pred_test,
         color='#d62728', linewidth=2, label='Test Prediction')
# Add train/test split line
split_date = test_data['date'].min() + pd.Timedelta(days=1)
plt.axvline(x=split_date, color='gray', linestyle=':', linewidth=2, label='Train/Test Split')

plt.title('AAL Stock Price Forecasting (Next Day Close)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 作业结果汇总
print("\n" + "="*60)
print("✅ 作业最终结果（可直接复制）：")
print("="*60)
print(f"任务类型：时间序列预测 (Forecasting)")
print(f"核心指标：MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | MAPE={test_mape:.2f}% | R²={test_r2:.4f}")
print("="*60)