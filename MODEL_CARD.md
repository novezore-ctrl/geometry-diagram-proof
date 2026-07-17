# Model Card — geometry_unet_pgdp5k_epoch5_320

## 模型概览

| 项目 | 内容 |
| --- | --- |
| 任务 | 几何图元六分类语义分割 |
| 架构 | 紧凑 U-Net，含深度可分离残差块 |
| 输入 | 1×3×320×320，RGB float32，范围 0–1 |
| 输出 | 1×6×320×320 logits |
| 类别 | background、solid_line、dashed_line、circle_or_ellipse、point、text |
| checkpoint | geometry_unet_pgdp5k_gpu.pt，epoch 5 |
| 浏览器格式 | ONNX opset 18，固定输入尺寸 |
| ONNX 本地导出位置 | public/models/geometry_unet_pgdp5k_epoch5_320.onnx（被 `.gitignore` 排除） |
| ONNX 大小 | 2,202,748 字节 |
| ONNX SHA-256 | 2548477e22be4965d38cb7d963f0504e05143616eef866746fa776219adb865a（本地已验证工件） |

## 预期用途

- 给裁剪后的印刷几何图生成线、曲线和点的像素候选；
- 作为确定性拓扑恢复和人工校正的上游感知模块；
- 在浏览器内通过 WebGPU 或 WASM 做本地推理；
- 用于教学、研究和工程原型中的失败分析。

模型输出必须经过结构后处理和人工复核。

## 不适用范围

- 不直接完成字母 OCR；
- 不理解任意题干逻辑；
- 不证明几何命题或计算角度；
- 不保证复杂手写页面、多个相邻图形或严重透视照片的正确结构；
- 不应作为无需复核的高风险决策系统；
- 不应把视觉坐标当作精确长度、角度或比例。

## 模型结构

编码通道为 24、48、96，瓶颈为 160，解码通道为 96、48、24。每个 Block 包含：

1. 3×3 卷积、BatchNorm、ReLU6；
2. 3×3 depthwise 卷积；
3. 1×1 pointwise 卷积；
4. 残差相加。

编码端最大池化，解码端双线性插值并与同尺度编码特征拼接，最后用 1×1 卷积输出六类 logits。

## 训练数据

主要真实数据为 PGDP5K：

- 总数 5,000；
- train 3,500；
- validation 500；
- test 1,000。

训练脚本支持把合成六分类图与真实 PGDP5K 转换数据混合。已记录的 GPU 训练使用 synthetic_v2、PGDP5K 命名训练/验证划分与 real-repeat=1。

PGDP5K 紧凑 geos 记录没有向当前转换器提供足够可靠的逐线型虚线字段，因此转换器把注释直线标为 solid_line。测试转换掩码也没有 text 像素。由此产生的重要限制是：当前正式测试不能验证 dashed_line 或 text。

数据不随此仓库分发。来源与条款边界见 data/README.md 和 THIRD_PARTY_NOTICES.md。

## 训练设置

已验证的 epoch-5 运行：

- 输入尺寸：320×320；
- CUDA 混合精度；
- batch size：16；
- data loader workers：2；
- CUDA allocator fraction：0.75；
- optimizer：AdamW，学习率 1e-3；
- loss：带类别权重的 CrossEntropyLoss；
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU；
- 完成日期：2026-07-16。

训练脚本在报告专用显存不超过 9 GiB 的 GPU 上拒绝 batch > 16。该限制来自本项目机器上的显存压力实测，目的是避免 WDDM 内存压力造成严重降速或整机卡顿；它不是所有 GPU 的理论最优 batch。

## 正式测试结果

报告文件：ml/evaluation/pgdp5k_test_epoch5.json

测试配置：

- split：test；
- images：1,000；
- batch size：16；
- point tolerance：10 px；
- line support threshold：0.58；
- elapsed：15.557 s；
- peak allocated VRAM：1.521 GiB。

### 像素指标

| 类别/指标 | Precision | Recall | IoU |
| --- | ---: | ---: | ---: |
| background | 0.998697 | 0.984534 | 0.983271 |
| solid_line | 0.539047 | 0.928956 | 0.517705 |
| circle_or_ellipse | 0.560995 | 0.904029 | 0.529463 |
| point | 0.626496 | 0.889827 | 0.581398 |
| dashed_line | 0 | 0 | 0；测试真值像素为 0 |
| text | 0 | 0 | 0；测试真值像素为 0 |

- pixel accuracy：0.983167；
- mean IoU over present classes：0.652959。

### 掩码派生拓扑代理

| 指标 | Precision | Recall |
| --- | ---: | ---: |
| 点 | 0.895616 | 0.975170 |
| 连接边 | 0.749347 | 0.933475 |

exact mask-derived graph rate：0.301。

## 评估边界

拓扑指标不是对 PGDP5K 原始关系标注的直接评分。评估代码分别从预测掩码和真值掩码提取点线图，再在 320×320 坐标中匹配：

- 点的最大距离容差为 10 像素；
- 线支持阈值为 0.58；
- 点匹配后才比较连接边；
- 点误检/漏检和边误检/漏检均为 0 才计入 exact graph。

这是一种可复现的产品导向代理，但不能替代原始关系标注或人工图审计。

图2的 7 点、10 线、0 箭头、4 个线内位置结果也不属于上述通用测试指标。该案例使用严格触发的题干模板，把模型候选映射到已知题干拓扑；A–G 不是 OCR 输出。

## 已知失效模式

- 印刷字母、手写笔迹和角标进入 point 类，形成假点；
- 点对线支持后处理会把假点放大为多条假边；
- 小点或弱线可能漏检；
- 箭头尖端可能像点；当前箭头仍靠后处理与人工确认；
- 统一缩放到 320×320 后，小字符和细标记信息不足；
- 多图同页、不正确选区和严重旋转/透视会显著恶化结果；
- 高 edge recall 伴随较低 edge precision，说明假连接仍是首要问题。

## 浏览器部署

app/mobileInference.ts 使用 onnxruntime-web 1.27.0：

- 兼容且处于安全上下文的 Windows/Android 浏览器优先 WebGPU；
- 失败时回退 WASM；
- iPhone/iPad Safari 使用单线程 CPU/WASM；
- Apple Neural Engine 不在 Web 版本的声明范围；
- 当使用者自行放置有权使用的 ONNX 工件时，模型从同源 `public/models/` 加载；
- 运行库从固定版本 CDN 按平台加载；
- 题图像素和张量不上传到 CDN 或推理服务。

记录的 Windows 图2回归时延为 WebGPU 1,190 ms、强制单线程 WASM 985 ms。它们是特定机器和案例的单次记录，不应当外推成所有设备基准。

## 监控与后续改进

发布新权重前至少需要：

1. 保留完整像素混淆矩阵；
2. 报告点/边代理指标与 exact graph；
3. 对最高错误样本做人工审计；
4. 跑通语义、坐标和图2回归；
5. 在目标 Windows 与 Apple 设备上重新验收；
6. 单独报告未出现在测试真值中的类别；
7. 检查数据与权重再分发条款。
