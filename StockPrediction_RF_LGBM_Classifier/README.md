# StockPrediction_RF_LGBM_Classifier - 股票涨跌预测系统

## 📋 项目简介

本项目是一个基于机器学习算法的股票涨跌预测系统，专注于使用**随机森林(RandomForest)**和**LightGBM**分类器预测股票次日的涨跌情况。项目采用滚动窗口训练方法，支持批量处理多只股票数据，并提供完整的模型训练、调参和评估流程。

## 🎯 项目目标

- 构建一个稳健的股票涨跌预测系统
- 实现批量处理多只股票数据的自动化训练流程
- 使用滚动窗口方法进行时间序列数据划分
- 比较RandomForest和LightGBM两种算法的性能
- 提供完整的模型训练、评估和保存功能

## 📊 数据集

**数据位置**: `../Stock_predict_analytics_csv/` 目录下的CSV文件

**数据格式要求**:
每个CSV文件应包含以下核心字段：
- `date`: 日期（YYYY-MM-DD格式）
- `open`, `high`, `low`, `close`: 开盘价、最高价、最低价、收盘价
- `volume`: 成交量
- `RSI_14`: 14日相对强弱指数
- `MACD`: 移动平均收敛散度
- `BB_width`: 布林带宽度
- `volatility_20`: 20日波动率
- `daily_return`: 日收益率

**目标变量**: `target` - 次日涨跌标签（1=涨，0=跌）

## 🏗️ 项目结构

```
StockPrediction_RF_LGBM_Classifier/
├── README.md                          # 项目说明文档
├── main.py                           # 主程序入口
├── data/
│   └── preprocess_data.py           # 数据加载和预处理模块
└── model/
    ├── train_and_tune.py           # 模型训练、调参和评估
    ├── rolling_window_train.py     # 滚动窗口划分逻辑
    └── saved_models_batch/         # 训练结果和模型保存目录
        ├── Stock A/               # 每只股票的独立结果目录
        ├── Stock AAL/
        └── Stock AAP/
```

## 🛠️ 技术栈

- **编程语言**: Python 3.7+
- **核心库**:
  - `scikit-learn`: RandomForest分类器、网格搜索、评估指标
  - `lightgbm`: LightGBM分类器
  - `pandas`, `numpy`: 数据处理
  - `joblib`: 模型序列化保存
  - `tqdm`: 训练进度条显示
- **算法**: 随机森林(RandomForest)、LightGBM梯度提升树

## 📈 核心算法与方法

### 1. 数据预处理 (`data/preprocess_data.py`)
- **数据加载**: 自动扫描并加载指定目录下的所有CSV文件
- **日期处理**: 转换为datetime格式，按日期升序排列
- **目标构造**: 基于次日收益率创建二分类标签（涨/跌）
- **特征选择**: 使用10个核心技术指标作为模型输入特征

### 2. 滚动窗口划分 (`model/rolling_window_train.py`)
采用时间序列友好的滚动窗口方法，确保训练数据在测试数据之前：
- **训练窗口**: 36个月历史数据
- **验证窗口**: 1个月数据（用于模型调参）
- **测试窗口**: 1天数据（预测次日涨跌）
- **滚动步长**: 1天（每日滚动预测）

### 3. 模型训练与调参 (`model/train_and_tune.py`)
对每个滚动窗口执行：
- **特征标准化**: 使用StandardScaler进行数据标准化
- **网格搜索**: 为每个模型搜索最优超参数
- **交叉验证**: 5折交叉验证选择最佳模型
- **双模型对比**: 同时训练RandomForest和LightGBM

### 4. 超参数搜索范围

**RandomForest**:
```python
{
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
```

**LightGBM**:
```python
{
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [15, 31]
}
```

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
cd CDS524_group_project

# 安装依赖
pip install pandas numpy scikit-learn lightgbm joblib tqdm python-dateutil
```

### 数据准备

1. 将股票数据CSV文件放入 `Stock_predict_analytics_csv/` 目录
2. 确保CSV文件包含必要的特征列（见"数据集"部分）

### 运行训练

```bash
# 进入项目目录
cd StockPrediction_RF_LGBM_Classifier

# 运行主程序
python main.py
```

### 程序输出

程序运行时将显示：
1. ✅ 数据目录验证
2. ✅ 发现的所有CSV文件列表
3. 📊 每只股票的滚动窗口训练进度
4. ⏱️ 每个窗口的训练详情（验证集准确率、F1分数）
5. 💾 模型保存路径信息
6. 📈 最终评分汇总（平均准确率、F1分数）

## 📊 评估指标

系统使用以下指标评估模型性能：

| 指标 | 说明 | 应用场景 |
|------|------|----------|
| **准确率** | 正确预测的比例 | 验证集和测试集评估 |
| **F1分数** | 精确率和召回率的调和平均 | 类别不平衡时的综合评估 |
| **参数优化** | 网格搜索最佳参数 | 模型调参过程 |

## 💾 结果保存

训练完成后，结果保存在 `model/saved_models_batch/` 目录下，每个股票有独立子目录：

```
saved_models_batch/Stock A/
├── all_results.json          # 所有窗口的详细结果
├── last_scaler.joblib        # 最后一轮的标准化器
├── RandomForest_final.joblib # RandomForest最终模型
├── LightGBM_final.joblib     # LightGBM最终模型
├── RandomForest_window_1.joblib # 第1个窗口的RandomForest模型
├── LightGBM_window_1.joblib    # 第1个窗口的LightGBM模型
└── ...                      # 其他窗口模型文件
```

**`all_results.json` 内容结构**:
```json
{
  "all_results": {
    "RandomForest": [...],    # 所有窗口的RandomForest结果
    "LightGBM": [...]         # 所有窗口的LightGBM结果
  },
  "total_windows": 260,       # 总窗口数
  "csv_file": "文件路径"       # 源数据文件路径
}
```

## 🔧 代码模块详解

### 主程序 (`main.py`)
- **功能**: 程序入口，协调整个训练流程
- **特点**: 批量处理所有CSV文件，支持增量训练模式
- **日志**: 自动记录训练过程到日志文件

### 数据预处理模块 (`data/preprocess_data.py`)
- **`load_and_preprocess_data()`**: 加载单个CSV并完成预处理
- **`get_all_csv_files()`**: 获取目录下所有CSV文件路径
- **异常处理**: 完善的错误检查和用户提示

### 模型训练模块 (`model/train_and_tune.py`)
- **`train_model_for_csv()`**: 单个CSV文件的完整训练流程
- **`batch_train_all_csv()`**: 批量训练所有CSV文件
- **`Logger`类**: 自定义日志系统，同时输出到控制台和文件
- **进度显示**: 使用tqdm显示训练进度，实时更新评分

### 滚动窗口模块 (`model/rolling_window_train.py`)
- **`get_window_splits()`**: 生成器函数，逐个产生窗口划分
- **时间处理**: 使用`dateutil.relativedelta`进行月份计算
- **数据划分**: 确保时间序列的时序正确性

## 📝 使用示例

### 基本使用
```python
# 导入必要的模块
from data.preprocess_data import load_and_preprocess_data, get_all_csv_files
from model.train_and_tune import train_model_for_csv

# 加载数据
df, X, y, dates, feature_cols = load_and_preprocess_data("path/to/stock.csv")

# 训练模型
results, save_dir = train_model_for_csv("path/to/stock.csv", "output/dir")
```

### 自定义训练参数
如需修改训练参数，可编辑 `model/train_and_tune.py` 中的以下部分：

1. **修改模型参数网格**（第60-80行）
2. **调整滚动窗口参数**（`model/rolling_window_train.py` 第15-17行）
3. **更改特征列**（`data/preprocess_data.py` 第33-35行）

## ⚡ 性能优化

1. **并行计算**: 网格搜索使用 `n_jobs=-1` 自动利用所有CPU核心
2. **内存管理**: 使用生成器逐窗口处理，避免一次性加载所有数据
3. **进度反馈**: 实时显示训练进度和当前窗口评分
4. **日志系统**: 详细记录训练过程，便于调试和复现

## 🔍 结果分析

训练完成后，可以通过以下方式分析结果：

1. **查看评分汇总**: 程序最后会输出每个模型的平均评分
2. **分析详细结果**: 查看 `all_results.json` 中的每个窗口表现
3. **模型比较**: 比较RandomForest和LightGBM在不同窗口的表现
4. **时间分析**: 观察模型在不同时间段的表现稳定性

## ⚠️ 注意事项

1. **数据质量**: 确保CSV文件包含所有必要的特征列
2. **时间顺序**: 数据必须按日期升序排列
3. **内存需求**: 保存所有窗口模型需要较大磁盘空间
4. **计算时间**: 滚动窗口训练较耗时，建议在性能较好的机器上运行