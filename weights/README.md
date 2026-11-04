# 推理权重

将训练完成的 plant-seg-lab `best_inference.pth` 或 `best.pth` 放在此目录。

推荐：`weights/best_inference.pth`。这是用于推理的完整分割模型权重，不是 ImageNet 骨干初始化权重；推理不需要额外下载预训练骨干。

分割权重由用户上传，不提交 Git。也可以使用 `--weights` 指向任意其他路径。

颜色分类器放在 `color/xgboost/`、`color/svm/`、`color/random_forest/`，与分割权重分开。
参见 [颜色分类说明](color/README.md)。

XGBoost、SVM、随机森林颜色模型均由用户手动上传，不随 Git 提交。目录说明见 color/xgboost/README.md。
