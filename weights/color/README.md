# 训练好的颜色分类器

按 xgboost/、svm/、random_forest/ 分目录存放自己的可信 pickle。
要求严格使用 xgboostcolor 的14维特征，默认0枯黄、1绿色；SVM保留标准化Pipeline。
参见仓库README的两阶段推理说明。这里不放分割模型或ImageNet初始化权重。

不包含实际模型文件。手动上传位置见 [XGBoost 目录说明](xgboost/README.md)。保留的 JSON 仅为来源与报告；21维模型不兼容当前14维推理。
