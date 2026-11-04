# XGBoost 颜色模型（手动上传）

实际 .pkl 文件不随 Git 提交。请将本地的 xgboost_stage2_model.pkl 上传到：

```text
/root/gpufree-data/plant-seg-inference/weights/color/xgboost/xgboost_stage2_model.pkl
```

源文件：[xgboostcolor/artifacts/xgboost_stage2_model.pkl](https://github.com/xuanpluspro/xgboostcolor/blob/f02507bdc00d3644e61866d15fbdeea1000f407d/artifacts/xgboost_stage2_model.pkl)。

推理参数：
```bash
--color-model weights/color/xgboost/xgboost_stage2_model.pkl --trust-color-model
```

现有接口要求14维特征 R,G,B,H,S,a,b,GE,M,YE,Cb,Cr,I,Q，默认1绿色、0枯黄；result3的21维模型不能用于当前接口。SVM和随机森林同样自行上传。模型依赖版本兼容性需实际加载验证，不重新训练、不自动修补旧pickle。

training_report*.json 和 models_manifest.json 保留来源记录；清单中的模型路径为手动上传建议位置，不代表仓库包含模型。报告不代表黑麦草精度。只对可信pickle启用 --trust-color-model。

普通删除不清除 Git 历史中的旧模型对象；网络受影响时可 shallow clone 最新快照到新的目录，不要覆盖含本地数据/权重的旧目录。
