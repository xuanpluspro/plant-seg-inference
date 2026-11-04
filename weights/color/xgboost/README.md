# 已训练 XGBoost 颜色分类器

以下文件从 [xgboostcolor](https://github.com/xuanpluspro/xgboostcolor/tree/f02507bdc00d3644e61866d15fbdeea1000f407d) 原样同步，未重新训练、未反序列化或修改模型。

| 本目录文件 | 源路径 | 用途 |
|---|---|---|
| xgboost_stage2_model.pkl | artifacts/xgboost_stage2_model.pkl | 当前两阶段命令使用的颜色分类器候选，14 维接口 |
| result1/xgboost_model.pkl | result1/xgboost_model.pkl | 历史14维版本 |
| result2/xgboost_model.pkl | result2/xgboost_model.pkl | 与 result1 内容相同，保留源目录对应关系 |
| result3/xgboost_21d_model.pkl | result3/xgboost_21d_model.pkl | 21维归档版本，不能用于当前14维推理接口 |

对应 training_report*.json 同步保留，报告仅代表源实验，不代表黑麦草上的精度。来源、Git blob SHA、字节数及 SHA256 记录在 models_manifest.json。

当前接口特征顺序为 R,G,B,H,S,a,b,GE,M,YE,Cb,Cr,I,Q，默认类别 1=green、0=senescent。启动时仍检查 n_features_in_、classes_ 和 predict_proba；文件名不代替检查。旧 pickle 可能与当前 XGBoost/sklearn 版本不兼容，需要匹配原训练环境；不自动补写未知属性。

```bash
python -u predict.py --weights weights/best_inference.pth --input data/ck1 \
  --output outputs/ck1color --mode full \
  --color-model weights/color/xgboost/xgboost_stage2_model.pkl \
  --trust-color-model --save-color-mask
```

使用前仍需安装颜色分类依赖，见主 README。分割权重需自行放入 weights/，不包含在这些颜色模型里。不会自动启用颜色分类；必须显式指定 --color-model。
