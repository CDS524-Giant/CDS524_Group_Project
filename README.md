# CDS524 股票预测项目 (Stock Price Prediction)

## 📋 项目简介

本项目是CDS524课程的综合分组项目，旨在构建一个完整的股票预测分析系统。我们收集了505家美国上市公司的历史股票数据，采用**分类**和**回归**两种机器学习方法，分别预测股票的涨跌方向和具体价格水平。该项目不仅实现了大规模数据处理，还提供了算法比较、性能评估和结果可视化等完整功能。

## 🎯 项目目标

- **大规模数据处理**: 自动化处理505家公司的股票历史数据
- **多维度预测**: 同时提供涨跌分类预测和价格回归预测
- **算法比较**: 在相同数据集上对比不同机器学习算法的性能
- **时间序列建模**: 采用滚动窗口方法处理金融时间序列特性
- **完整工作流**: 从数据预处理到模型评估的端到端解决方案
- **可复现性**: 提供详细文档和代码，便于学术研究和应用

## 📊 数据集概述

### 数据来源
- **数据位置**: `Stock_predict_analytics_csv/` 目录
- **公司数量**: 505家美国上市公司
- **时间跨度**: 多年历史数据（因公司而异）
- **总文件数**: 505个CSV文件

### 数据字段
每个股票CSV文件包含完整的金融数据：

**基础价格数据**:
- `date`: 交易日期（YYYY-MM-DD格式）
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

**技术指标** (预计算):
- `RSI_14`: 14日相对强弱指数
- `MACD`: 移动平均收敛散度
- `BB_width`: 布林带宽度
- `volatility_20`: 20日波动率
- `daily_return`: 日收益率

### 数据规模
- **总记录数**: 数百万条交易记录
- **特征维度**: 10+个技术指标
- **预测目标**:
  - 分类：次日涨跌标签（1=上涨，0=下跌）
  - 回归：次日收盘价预测

## 🏗️ 项目架构

```
CDS524_group_project/
├── README.md                                    # 项目总说明文档
├── Stock_predict_analytics_csv/                 # 原始股票数据（505家公司）
│   ├── Stock A.csv
│   ├── Stock AAPL.csv
│   ├── Stock MSFT.csv
│   └── ... (共505个文件)
├── StockPrediction_RF_LGBM_Classifier/          # 分类预测模块
│   ├── README.md
│   ├── main.py                                 # 分类主程序
│   ├── data/
│   │   └── preprocess_data.py                  # 分类数据预处理
│   └── model/
│       ├── train_and_tune.py                   # 模型训练调参
│       ├── rolling_window_train.py             # 滚动窗口逻辑
│       └── saved_models_batch/                 # 分类模型结果
├── StockPrediction_LGBM_XGBoost_Regression/    # 回归预测模块
│   ├── README.md
│   ├── data/
│   │   ├── train_all_stocks.py                # 回归主训练脚本
│   │   ├── data_preprocessing.py              # 回归数据预处理
│   │   ├── xgboost_model.py                   # XGBoost实现
│   │   ├── lightgbm_model.py                  # LightGBM实现
│   │   ├── rolling_window.py                  # 滚动窗口工具
│   │   ├── analyze_data.py                    # 数据分析工具
│   │   ├── calculate_accuracy.py              # 评估工具
│   │   ├── test_models.py                     # 模型测试
│   │   └── stock_prediction.py                # 单股票预测
│   └── prediction_results/                    # 回归预测结果
└── prediction_results/                         # 全局结果目录
```

## 🛠️ 技术栈

### 编程语言
- **Python 3.7+**: 主要开发语言

### 核心依赖库
- **数据处理**: `pandas`, `numpy`
- **机器学习**: `scikit-learn`, `xgboost`, `lightgbm`
- **模型持久化**: `joblib`
- **进度显示**: `tqdm`
- **可视化**: `matplotlib`, `seaborn`

### 机器学习算法

#### 分类模块 (StockPrediction_RF_LGBM_Classifier)
- **随机森林 (RandomForest)**: 集成学习，适合高维特征
- **LightGBM**: 梯度提升决策树，针对大数据优化

#### 回归模块 (StockPrediction_LGBM_XGBoost_Regression)
- **XGBoost**: 极端梯度提升，强大的回归性能
- **LightGBM**: 高效梯度提升框架

## 📈 核心方法论

### 时间序列建模策略
两个模块都采用**滚动窗口 (Rolling Window)** 方法：

1. **训练窗口**: 使用历史数据训练模型
2. **验证窗口**: 调参和模型选择
3. **测试窗口**: 评估预测性能
4. **滚动步长**: 逐日滚动，确保时间顺序

### 分类预测流程
1. **数据预处理**: 构造涨跌标签，特征选择
2. **超参数调优**: 网格搜索最优参数
3. **模型训练**: RandomForest vs LightGBM对比
4. **性能评估**: 准确率、精确率、召回率、F1分数
5. **结果保存**: 每个股票的预测结果和模型

### 回归预测流程
1. **特征工程**: 价格数据和技术指标
2. **滚动窗口训练**: 多窗口性能平均
3. **模型比较**: XGBoost vs LightGBM
4. **误差分析**: RMSE、MAE、R²等指标
5. **预测输出**: 实际价格 vs 预测价格对比

## 🚀 快速开始

### 环境要求
- Python 3.7 或更高版本
- 推荐内存: 8GB+ RAM
- 磁盘空间: 2GB+ (数据和模型)

### 安装依赖
```bash
# 创建虚拟环境（推荐）
conda create -n stock_prediction python=3.8
conda activate stock_prediction

# 安装核心依赖
pip install pandas numpy scikit-learn xgboost lightgbm joblib tqdm matplotlib seaborn
```

### 运行完整预测流程

#### 1. 分类预测（涨跌方向）
```bash
cd StockPrediction_RF_LGBM_Classifier
python main.py
```

#### 2. 回归预测（具体价格）
```bash
python StockPrediction_LGBM_XGBoost_Regression/data/train_all_stocks.py
```

## 📊 输出结果

### 分类模块输出
- **模型文件**: `saved_models_batch/{股票名称}/` 目录下的.pkl文件
- **性能报告**: 控制台输出的准确率、精确率等指标
- **预测结果**: 每个股票的涨跌预测标签

### 回归模块输出
- **预测CSV**: `prediction_results/prediction_results_{股票名称}.csv`
- **性能指标**: RMSE、MAE、R²分数
- **算法对比**: XGBoost vs LightGBM性能比较

### 示例输出文件结构
```
prediction_results_Stock_AAPL.csv
├── date: 预测日期
├── actual: 实际收盘价
├── xgboost_pred: XGBoost预测价
└── lightgbm_pred: LightGBM预测价
```

## 🔬 实验设计

### 数据分割策略
- **时间序列保护**: 严格按时间顺序分割
- **无未来数据泄露**: 训练数据始终在测试数据之前
- **滚动验证**: 多轮交叉验证确保稳健性

### 评估指标

#### 分类任务
- **准确率 (Accuracy)**: 整体预测正确率
- **精确率 (Precision)**: 正类预测准确性
- **召回率 (Recall)**: 正类识别完整性
- **F1分数**: 精确率和召回率的调和平均

#### 回归任务
- **RMSE**: 均方根误差，惩罚大误差
- **MAE**: 平均绝对误差，平均预测误差
- **R²**: 决定系数，模型解释方差比例

### 基准比较
- **算法对比**: RandomForest vs LightGBM (分类)
- **算法对比**: XGBoost vs LightGBM (回归)
- **股票表现**: 不同股票上的模型性能差异

## 💡 项目亮点

### ✨ 技术优势
- **大规模处理**: 自动化处理505家公司数据
- **双重验证**: 分类+回归互补验证
- **时间序列友好**: 专业的滚动窗口实现
- **算法全面**: 主流ML算法的完整比较
- **代码质量**: 模块化设计，易于维护和扩展

### 📈 研究价值
- **金融AI应用**: 机器学习在量化金融中的实践
- **算法评估**: 在真实金融数据上的性能对比
- **特征重要性**: 技术指标的有效性分析
- **可扩展性**: 为更大规模金融预测提供框架

## 🔧 故障排除

### 常见问题
1. **内存不足**: 处理大数据时考虑分批处理
2. **运行时间长**: 完整训练可能需要数小时
3. **路径错误**: 确保从项目根目录运行脚本
4. **依赖冲突**: 使用虚拟环境隔离依赖

### 性能优化建议
- **数据采样**: 对超大数据集进行采样测试
- **并行处理**: 考虑多进程处理多个股票
- **特征选择**: 减少不重要特征以提升速度
- **模型简化**: 调整超参数以平衡精度和速度