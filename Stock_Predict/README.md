# 股票价格预测项目 (Stock Price Prediction)

## 📋 项目简介

本项目是 CDS524 课程的分组项目，目标是使用机器学习算法预测美国航空（American Airlines, AAL）的股票每日收盘价。通过提取历史股票数据的特征，建立线性回归模型，实现短期股票价格的预测。

## 🎯 项目目标

- 利用历史股票数据进行特征工程
- 建立回归模型预测次日股票收盘价
- 评估模型的预测性能
- 分析不同特征对预测的影响

## 📊 数据集

**数据源**: `AAL_2013-03-11_to_2018-02-07_all_data.csv`

- **股票代码**: AAL（美国航空）
- **数据跨度**: 2013年3月11日 - 2018年2月7日
- **数据周期**: 日线数据
- **包含字段**: 日期(date)、开盘价(open)、高价(high)、低价(low)、收盘价(close)、成交量(volume)等

## 🛠️ 项目结构

```
Stock_Predict/
├── README.md                                    # 项目说明文档
├── EvaluationMetrics.py                        # 主程序（数据处理、建模、评估）
└── AAL_2013-03-11_to_2018-02-07_all_data.csv  # 股票历史数据
```

## 📈 主要方法

### 1. 数据预处理
- 读取CSV文件中的股票数据
- 时间戳标准化（转换为pandas datetime格式）
- 按日期排序，处理缺失值
- 构建预测目标：次日收盘价

### 2. 特征工程
项目使用以下特征进行模型训练：
- **滞后特征** (Lagged Features)
  - `close_lag_1` 到 `close_lag_5`: 过去1-5天的收盘价
  
- **技术指标** (Technical Indicators)
  - `SMA_5`: 5日简单移动平均线
  - `SMA_10`: 10日简单移动平均线
  - `RSI_14`: 相对强弱指数（如果可用）
  - `MACD`: 移动平均线收敛散度（如果可用）
  - `volatility_20`: 20日波动率（如果可用）

### 3. 模型训练
- **算法**: 线性回归 (Linear Regression)
- **数据分割**: 80% 训练集，20% 测试集
- **样本**: 约90,000+条历史数据

### 4. 模型评估
采用多个评估指标衡量模型性能：

| 指标 | 说明 |
|------|------|
| **MAE** | 平均绝对误差 (Mean Absolute Error) - 预测价格的平均偏离金额 |
| **RMSE** | 均方根误差 (Root Mean Square Error) - 惩罚较大误差 |
| **MAPE** | 平均绝对百分比误差 (Mean Absolute Percentage Error) - 相对误差百分比 |
| **R²** | 决定系数 (R-squared) - 模型拟合优度，范围[0,1] |

## 🚀 使用方法

### 环境要求
```
Python 3.7+
pandas
numpy
scikit-learn
matplotlib
```

### 安装依赖
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 运行程序
```bash
python EvaluationMetrics.py
```

### 程序输出
程序会依次输出：
1. ✅ 发现的CSV文件列表
2. ✅ 数据读取状态及数据形状
3. ✅ 数据列名
4. ✅ 特征工程完成情况
5. 📊 训练集评估指标 (MAE, RMSE, MAPE, R²)
6. 📊 测试集评估指标
7. 📈 性能对比图表

## 📝 关键代码流程

```python
# 1. 自动扫描并加载CSV文件
df = pd.read_csv(selected_file)

# 2. 数据预处理
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['next_day_close'] = df['close'].shift(-1)

# 3. 特征工程
create_time_features(df, window=5)

# 4. 训练/测试分割
train_data, test_data = train_test_split(df, test_size=0.2)

# 5. 模型训练
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 预测与评估
y_pred = model.predict(X_test)
calculate_metrics(y_test, y_pred)
```

## 💡 项目特点

✨ **优势**:
- 完整的机器学习流程（数据 → 特征 → 模型 → 评估）
- 自动化文件扫描，易于处理多个数据源
- 多维度评估指标，全面反映模型性能
- 中英文双语注释，易于理解

⚠️ **局限性**:
- 线性回归模型相对简单，可尝试更复杂的模型（如xgboost、LSTM等）
- 仅使用历史价格数据，未考虑外部因素（新闻、宏观经济指标等）
- 假设历史规律可预测未来，存在市场变化风险

## 🔄 后续改进方向

1. **模型优化**
   - 尝试非线性回归（Random Forest, XGBoost, SVM等）
   - 实现深度学习模型（LSTM, GRU等）

2. **特征增强**
   - 加入更多技术指标（布林带、KDJ等）
   - 考虑市场情绪指标
   - 添加外部数据（VIX恐慌指数、经济数据等）

3. **模型评估**
   - 时间序列交叉验证（Time Series CV）
   - 回测分析（Backtesting）
   - 风险权衡分析

4. **实用化**
   - 实时价格预测API
   - 可视化仪表板
   - 自动交易信号生成

## 👥 项目成员

CDS524 Group Project Team

## 📄 许可证

内部使用，不对外公开

---

**最后更新**: 2024年

如有问题或建议，欢迎提出 Issue 或 Pull Request！
