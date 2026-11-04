# 训练好的颜色分类器

按 xgboost/、svm/、random_forest/ 分目录存放自己的可信 pickle。
要求严格使用 xgboostcolor 的14维特征，默认0枯黄、1绿色；SVM保留标准化Pipeline。
参见仓库README的两阶段推理说明。这里不放分割模型或ImageNet初始化权重。

已内置 xgboostcolor 的 XGBoost 模型及历史版本，详见 [XGBoost 模型清单](xgboost/README.md)。21维版本仅归档，不兼容当前14维推理。
