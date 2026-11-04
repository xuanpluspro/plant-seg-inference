# plant-seg-inference

独立的植物语义分割推理工具，以 [plant-seg-lab](https://github.com/xuanpluspro/plant-seg-lab) 为模型蓝本。不包含训练入口，不需要同时克隆训练仓库。

## 默认行为

- GPU 单图、整图推理；不缩放、不切块；默认开启 FP16 autocast。模型内原有的 FP32 特征交互保持不变。
- 图片宽高只向右/下补到 32 的倍数（最小 64），输出裁回原始尺寸。3072×3072 不补边。
- 默认保留最大前景连通域（8 邻域、按像素面积），去除与主体分离的小区域；整图/滑窗均在最终整张掩膜上处理。
- 默认仅保存 `原图名_overlay.jpg`，植物前景以绿色半透明叠加，JPG quality=90。原图不改动。
- 所有结果集中在一个输出目录；`meta/` 内保存运行参数、日志和逐图耗时。不会在仓库外另建日志目录。
- 显存不足明确退出；不自动缩图、转滑窗或改用 CPU。CPU 版 PyTorch 不能执行推理。
- 显式滑窗模式：512 窗口、128 重叠，逐块推理，在 CPU 内存按正三角权重融合 float32 概率，然后生成掩膜。无 CPU 模型前向计算。

## 安装

GPUFree 镜像已有 torch/torchvision 时保留现有安装，不重复下载。先检查：

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

按需逐个安装辅助依赖（上交镜像）：

```bash
uv pip install --python "$(command -v python)" numpy -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" Pillow -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" scipy -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

本地 Windows 应在 GPU 可用、torch 与 torchvision 匹配的环境中运行；`requirements.txt` 不会安装或替换 torch。

## GPUFree 整图推理

```bash
cd /root/gpufree-data
git clone https://github.com/xuanpluspro/plant-seg-inference.git
cd plant-seg-inference
```

`weights/` 存放训练完成的分割模型权重，与训练时的 ImageNet 骨干初始化权重区分。

将训练权重放在 `weights/best_inference.pth`，图片放在 `data/ck1`，或用绝对路径指向现有文件。不需要 ImageNet 预训练文件。

```bash
python -u predict.py \
  --weights weights/best_inference.pth \
  --input data/ck1 \
  --output outputs/ck1visual \
  --mode full
```

默认 batch=1。整图 3072×3072 对显存要求高，48GB 也需实测；8GB laptop 曾出现 OOM。降低 JPG 质量只减少磁盘占用，不减少网络前向显存。

本地 PowerShell（仓库目录内）：

```powershell
python predict.py --weights "F:\中期\三数据集一起训练的二分类模型\best_inference.pth" --input "F:\中期\三数据集一起训练的二分类模型\ck1" --output "F:\中期\三数据集一起训练的二分类模型\ck1visual" --mode full
```

若省略 `--output`，默认在输入同级创建 `<输入名>visual`，例如 `ck1visual`。支持单张图或递归读取文件夹，保留子目录；同目录同 stem 的 jpg/png 会提前报命名冲突，不生成双扩展名。

## 滑窗、续跑和可选输出

显存不足后，手动选择新的输出目录运行滑窗，不复用整图的运行配置：

```bash
python -u predict.py --weights weights/best_inference.pth --input data/ck1 \
  --output outputs/ck1visual_tile --mode tile --tile-size 512 --overlap 128
```

相同命令末尾加 `--resume` 可跳过已完成且 size/mtime 未变的图片。会核对权重 SHA256、输入路径、源模型版本和全部影响输出的推理设置。未完成图片重新处理。更换权重、模式或叠加参数需新输出目录。输入内容变更但刻意保留 size/mtime 不在自动检测范围内。续跑不要手动修改已完成结果。

`--alpha 0.4` 控制叠加浓度，`--jpg-quality 90` 控制 JPEG 压缩质量，`--no-amp` 关闭混合精度。

默认按类别 argmax，背景为 0；二分类可显式加 `--threshold 0.5`，规则为前景概率严格大于阈值。多分类使用固定调色板，类别 1 为绿色。

可选 `--save-mask` 输出 extras/ 下 PNG（二分类 0/255，多分类索引）；`--save-probabilities` 输出 C×H×W float32 NPY。默认两者关闭。滑窗与整图上下文、GroupNorm 统计不同，不能保证数值或掩膜一致。

输出结构示例：

```text
outputs/ck1visual/
  523_ck1_BLN_overlay.jpg
  meta/console.log
  meta/run.json
  meta/per_image.csv
```

`run.json` 每完成一图即原子更新；CSV 在批次正常结束后导出。耗时含预处理、GPU 推理与结果回传，不含模型加载及图片编码；保存耗时单独记录。无 GT 时不会报告精度、IoU 等评价指标。

## 旧训练权重瘦身

支持含 optimizer 等训练状态的完整 v4 checkpoint，以及已瘦身的 best_inference.pth。保留 format/model_config/完整模型权重及元数据；严格加载，不自动忽略缺失参数。不支持只有 state_dict 且无模型配置的文件，也不猜测旧版架构。

```bash
python slim_checkpoint.py --weights weights/best.pth --output weights/best_inference.pth
```

使用 CPU 读取和保存，不训练、不构建模型、不改变权重精度。兼容中文路径，以文件对象读写；逐张量核对 dtype、shape 和数值后生成最终文件。原文件不覆盖。推理文件不能用于完整断点续训。

## 模型来源与验证边界

`model/` 从 UPSTREAM.json 标记的源提交原样复制，包含所需模块及原始来源/许可证文件。不会自动跟随训练仓库更新。更新模型源码时需核对版本和权重兼容性。

轻量检查：`python -m unittest discover -s tests -v`，只测试图像处理、滑窗概率融合、输出命名、续跑配置及模拟前向调度，不导入 torch，不运行真实神经网络。语法编译另行检查。真实 checkpoint 加载、CUDA/AMP、3072 整图显存及分割效果尚需在服务器验证；这些测试不构成 GPU 精度验证。

## 最大连通域后处理

默认启用 `--largest-component`，使用 8 邻域：水平、垂直、对角相邻的前景像素都视为连通。根据前景像素数保留最大区域，其余置为背景，再生成 overlay 和可选 mask。滑窗模式先融合整图概率、生成整图类别掩膜，再做一次全局连通域分析，不是每块各保留一个。

只按几何连通性和面积选择，不识别“哪株是目标黑麦草”：与主体断开的真实叶片也会删除，连到主体的杂草不会删除；若其他区域更大，会保留那个区域。不会自动闭运算、填洞或把断开的叶片连接起来。空前景保持全背景；面积相同时取按行扫描最先出现的区域。多分类按所有非零类的并集选取最大区域，保留区内原类别。

`--no-largest-component` 关闭过滤；`--connectivity 4` 改为只考虑水平/垂直相邻。网络权重和原始图片均不修改。可选 `--save-probabilities` 仍保存后处理前的网络概率，不会伪造过滤后的概率。

后处理在 CPU 数组上执行，不运行 CPU 神经网络。日志报告连通域数、保留前后像素数；`meta/run.json` 保存逐图统计，CSV 单独记录 `postprocess_seconds`。新增参数也参与续跑配置核对。

如果此前已经生成未过滤的结果，使用新输出目录重新运行（例如 `outputs/ck1visual_largest`），不要对旧结果直接 `--resume`，否则会因配置不同而拒绝续跑。为节省磁盘仍默认只输出 JPG overlay。

```bash
python -u predict.py --weights weights/best_inference.pth --input data/ck1 \
  --output outputs/ck1visual_largest --mode full --largest-component
```

## 覆盖同一结果目录

`--overwrite` 从头处理全部输入，原子替换同名输出图片，并重建本次运行记录。允许改变后处理设置；不能与 `--resume` 同时使用。不清空整个目录，不删除其他文件；未被本次输入覆盖的旧图片或未启用的额外输出仍会保留。运行过程中会暂时同时存在新旧结果，正常结束后当前输入对应结果均已更新；中断后用相同设置加 `--resume` 完成剩余图片（去掉 `--overwrite`）。

```bash
python -u predict.py --weights weights/best_inference.pth --input data/ck1 \
  --output outputs/ck1visual --mode full --largest-component --overwrite
```

旧说明中的“使用新目录”是默认保护行为；明确加 `--overwrite` 即可覆盖原目录，无需先删图片。

## 分割后区分绿叶／枯黄叶并统计枯叶率

参考 SegVeg 的两阶段思路：先用现有二分类分割模型获得植物前景，再对**原图 RGB**中的保留前景提取颜色特征，交给已经训练好的 XGBoost、SVM 或 RandomForest 分类器。颜色分类发生在最大连通域筛选之后，不会把已删除区域重新计入植物。不开启 `--color-model` 时，原分割行为保持不变。

特征直接沿用 [xgboostcolor](https://github.com/xuanpluspro/xgboostcolor) 提交 `f02507bdc00d3644e61866d15fbdeea1000f407d` 的 `vegetation_predictor.py`，顺序为 **R,G,B,H,S,a,b,GE,M,YE,Cb,Cr,I,Q**。RGB 为 0–255、H 为角度、S 为百分比；GE 是 RGB 均值，YCbCr 用 Pillow 转换，其余计算保持源代码一致。它是本仓库与该颜色仓库的特征契约，不声称与论文所有颜色转换细节逐位一致。只支持用这套 14 维原始特征训练的二分类模型，21 维模型不能混用；SVM 的标准化器必须和模型一起保存为 Pipeline，不能丢失训练预处理。

### 准备颜色分类器

仓库已内置从 xgboostcolor 原样同步的 XGBoost 模型，`git pull` 后无需另外下载。默认示例使用：

- `weights/color/xgboost/xgboost_stage2_model.pkl`
- `weights/color/svm/svm_stage2.pkl`
- `weights/color/random_forest/random_forest.pkl`

本仓库内置的 XGBoost 文件及报告见 `weights/color/xgboost/README.md`，未重新训练。SVM 和随机森林权重仍需自行准备。源颜色仓库的 `artifacts/xgboost_stage2_model.pkl`、14维 `result*/xgboost_model.pkl` 及 `stage2_pipeline.py` 训练的模型属于候选格式，实际可加载性取决于 pickle 与依赖版本。随机森林同样要求 14 维特征、`classes_`、`n_features_in_`、`predict_proba`；本工具不负责训练分类器。XGBoost、SVM、随机森林是三种可选分类器，一次运行使用一个，不自动投票。

按需逐个安装（不更换 torch）：

```bash
uv pip install --python "$(command -v python)" scikit-image -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" scikit-learn -i https://mirror.sjtu.edu.cn/pypi/web/simple
# 仅 XGBoost 模型需要：
uv pip install --python "$(command -v python)" xgboost -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

pickle 应使用训练时匹配的依赖版本；安装最新版不保证旧 pickle 可用。这里不会靠补写未知属性掩盖版本问题；旧模型加载或预测报错时，应使用匹配环境重新导出或训练。SVM 必须训练为 `probability=True`。分类器默认标签 `1=green`、`0=senescent`，通过 `classes_` 定位概率列，不假定绿色永远位于第二列；其他整数标签可显式传 `--color-green-label`、`--color-senescent-label`。

### 一条命令执行两阶段推理

```bash
python -u predict.py \
  --weights weights/best_inference.pth \
  --input data/ck1 \
  --output outputs/ck1color \
  --mode full \
  --color-model weights/color/xgboost/xgboost_stage2_model.pkl \
  --trust-color-model \
  --save-color-mask
```

将 `--color-model` 换为 SVM 或随机森林路径即可。pickle 可能执行代码，因此仅对自己可信的模型使用 `--trust-color-model`。启动时检查特征数、类别及概率接口；神经网络仍只在 CUDA 推理，颜色分类与特征计算在 CPU 数组上运行，不在 CPU 训练或执行分割网络。每批最多 `--color-chunk-size 65536` 个候选像素，避免一次展开 3072×3072×14 特征；SVM 对大量像素仍可能明显耗时。

默认绿色概率 `>= --color-threshold 0.5` 判为绿色，其余植物像素判为枯黄。背景不进入颜色分类。保留原图尺寸和坐标，不从绿色 overlay 或涂黑图中提取颜色。当前只接受二分类植物／背景分割模型，拒绝把多分类语义类别静默替换为颜色类别。

### 输出及统计口径

启用颜色分类后，原 `*_overlay.jpg` 改为绿叶绿色、枯黄叶黄色的叠加图，背景保持原图。`--save-color-mask` 额外保存 `extras/*_color_mask.png`，为 P 模式索引 PNG：0 背景（黑）、1 绿色（0,180,70）、2 枯黄（255,255,0）。原 `--save-mask` 仍是植物二分类掩膜；分割概率 NPY 也保持原含义，不是颜色分类概率。

统计集中在 `meta/color_metrics.csv`，每完成一张图片即原子更新，并保存在 `meta/run.json` 的逐图记录中：

| 字段 | 定义 |
|---|---|
| foreground_pixels | 连通域筛选后植物前景像素数 G+S |
| green_pixels / senescent_pixels | 绿色 G／枯黄 S 的像素数 |
| vegetation_fraction | (G+S)/整图像素数 |
| green_fraction | G/整图像素数 |
| senescent_fraction | S/整图像素数，SegVeg 风格 SF |
| senescent_ratio | S/(G+S)，你要的前景内枯叶率，0–1 |
| senescent_percent | 100×S/(G+S)，0–100百分数 |
| color_seconds | 该图颜色特征计算与分类耗时 |

空前景时计数和整图面积比例为 0，枯叶率为空（JSON 为 null），不会把“没有植物”误记成“枯叶率为零”。统计的是可见投影面积比例，不能代表叶片数量、真实三维叶面积或直接证明组织已死亡。分割前景若包含茎、穗等，比例也包含这些组织；要称为严格叶片枯叶率，需要前景本身只包含叶片。

原图光照、白平衡和物种变化会影响颜色分类；源仓库的测试分数不能直接视为黑麦草上的准确率。建议人工核查一批目标域绿／枯黄像素及最终面积比例。最大连通域过滤也可能删除断开的真实叶片，需检查实际保留范围。

颜色模型 SHA256、类别映射、阈值、分块大小、掩膜开关加入续跑配置核对。纯分割旧结果不能直接作为颜色结果跳过：请使用新输出目录，或显式 `--overwrite` 重算。颜色 CSV 只写本次运行清单中的记录；颜色阈值变更也必须重新运行。旧纯分割续跑配置保持兼容。

验证：新增轻量测试覆盖前景筛选、分块一致、空掩膜、绿色概率列、异常概率、P 模式调色板、CSV 空值与原子更新。不运行 CPU 神经网络。真实颜色 pickle 的跨版本加载、scikit-image 特征执行及端到端 CUDA 联合推理仍需在用户环境验证。
