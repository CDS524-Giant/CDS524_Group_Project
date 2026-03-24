# StockPrediction_LGBM_XGBoost_Regression

## 项目简介

这是一个基于XGBoost和LightGBM算法的股票价格回归预测系统。该项目专门设计用于预测股票的次日收盘价，通过机器学习技术分析历史股票数据中的模式和趋势。

## 主要功能

- **双模型预测**: 同时使用XGBoost和LightGBM进行价格预测
- **滚动窗口验证**: 采用时间序列友好的滚动窗口交叉验证方法
- **批量处理**: 支持自动处理多个股票数据集
- **特征工程**: 包含技术指标和日期特征的自动特征构建
- **性能评估**: 提供RMSE、MAE、R²等多维度评估指标
- **结果保存**: 为每个股票生成详细的预测结果CSV文件

## 技术栈

- **编程语言**: Python 3.7+
- **核心库**:
  - `pandas`, `numpy`: 数据处理和数值计算
  - `scikit-learn`: 数据预处理和评估指标
  - `xgboost`: 极端梯度提升回归模型
  - `lightgbm`: 轻量级梯度提升机器学习框架

## 项目结构

```
StockPrediction_LGBM_XGBoost_Regression/
├── README.md                          # 项目说明文档
├── data/                              # 核心代码目录
│   ├── train_all_stocks.py           # 主训练脚本
│   ├── data_preprocessing.py         # 数据加载和预处理
│   ├── xgboost_model.py              # XGBoost模型实现
│   ├── lightgbm_model.py             # LightGBM模型实现
│   ├── rolling_window.py             # 滚动窗口处理逻辑
│   ├── analyze_data.py               # 数据分析工具
│   ├── calculate_accuracy.py         # 准确度计算工具
│   ├── test_models.py                # 模型测试脚本
│   └── stock_prediction.py           # 单个股票预测
└── prediction_results/               # 预测结果输出目录（运行后生成）
```

## 数据要求

### 数据位置
项目依赖 `../Stock_predict_analytics_csv/` 目录下的股票数据文件

### 数据格式
每个CSV文件应包含以下字段：
- `date`: 交易日期（YYYY-MM-DD格式）
- `open`, `high`, `low`, `close`: OHLC价格数据
- `volume`: 成交量
- 技术指标字段（RSI、MACD等，可选）

### 目标变量
- `target`: 次日收盘价（系统自动生成）

## 安装和使用

### 环境要求
- Python 3.7 或更高版本
- 推荐使用虚拟环境

### 安装依赖
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

### 运行预测
```bash
# 从项目根目录运行
python StockPrediction_LGBM_XGBoost_Regression/data/train_all_stocks.py
```

## 核心算法

### XGBoost模型
- **目标**: 回归预测
- **损失函数**: 平方误差
- **默认参数**:
  - max_depth: 6
  - learning_rate: 0.1
  - n_estimators: 100
  - subsample: 0.8
  - colsample_bytree: 0.8

### LightGBM模型
- **目标**: 回归预测
- **指标**: RMSE
- **参数设置**: 与XGBoost保持一致以便公平比较

### 滚动窗口策略
- **训练窗口**: 使用历史数据进行模型训练
- **验证窗口**: 用于模型调参和早停
- **测试窗口**: 最终性能评估
- **时间序列保护**: 确保训练数据始终在测试数据之前

## 输出结果

### 控制台输出
- 每个股票的训练进度
- 滚动窗口性能指标
- 最终测试集表现对比
- 算法性能比较结果

### 文件输出
为每个股票生成 `prediction_results/prediction_results_{股票名称}.csv`，包含：
- `date`: 预测日期
- `actual`: 实际价格
- `xgboost_pred`: XGBoost预测价格
- `lightgbm_pred`: LightGBM预测价格

### 性能指标
- **RMSE (均方根误差)**: 衡量预测精度
- **MAE (平均绝对误差)**: 平均预测误差
- **R² (决定系数)**: 模型解释方差比例

## 使用示例

### 批量训练所有股票
```python
from data.train_all_stocks import main
main()
```

### 单个股票分析
```python
from data.stock_prediction import predict_single_stock
predict_single_stock('Stock_AAPL.csv')
```

## 模型评估

### 交叉验证
- 多轮滚动窗口验证
- 时间序列数据完整性保证
- 防止数据泄露

### 算法比较
系统自动比较XGBoost和LightGBM在每个股票上的表现，并输出哪个算法性能更优。

## 注意事项

1. **数据质量**: 确保输入数据完整且格式正确
2. **计算资源**: 处理大量股票数据需要足够的内存
3. **运行时间**: 完整训练所有股票可能需要较长时间
4. **路径依赖**: 脚本假设从项目根目录运行