# GeoStruct

> 人在回路的几何图结构化与端侧推理系统
>
> Human-in-the-loop geometry diagram parsing with a compact U-Net, deterministic topology extraction, and in-browser inference.

GeoStruct 把几何题照片中的图形转换为可校正、可追踪的结构化图：先用轻量 U-Net 分割线、曲线、点等图元，再用确定性规则生成点线拓扑，最后由用户修正名称、连接、箭头和“点在线段上的位置”，导出供多模态模型核对的原图与 geometry-diagram/v5 JSON。

这不是自动解题器，也不是通用几何 OCR。当前最可靠的用法是“模型给候选、规则保结构、用户做最终确认”，而不是直接把机器输出当作题目事实。

## 项目状态与能力边界

当前仓库已经包含：

- 320×320、六分类的轻量 GeometryUNet；
- 使用 PGDP5K 的 3,500 张训练图、500 张验证图，并完成 1,000 张测试图评估的 epoch-5 模型；
- 已完成 2,202,748 字节 ONNX 模型的本地导出与浏览器验证；公开仓库暂不再分发该权重，原因见下文许可说明；
- Windows 浏览器 WebGPU 与 CPU/WASM 回退路径；
- iPhone/iPad Safari 的单线程 CPU/WASM 路径；
- 图片选择、滚轮/双指缩放、拖动、90° 旋转和一键清除；
- 屏幕尺寸固定的覆盖层，放大原图时点、标签和线宽不会同步变粗；
- 点名、点位和连接关系的人工编辑；
- 箭头候选确认、线内连接和 2–10 等分点确认；
- 原图与结构说明分步交给 GPT 的 geometry-diagram/v5 分享流程；
- 真实失败案例、语义回归和浏览器渲染回归。

仍未完成或不能宣称完成的部分：

- 没有可靠的 A/B/C/D 字母分类器。通用回退中的 P1/P2… 是临时拓扑编号；
- 没有自动证明、角度求解或题干逻辑推理；
- 虚线和文字类别没有在当前 PGDP5K 转换测试集上得到有效验证；
- 对复杂拍照页面、手写笔迹、角标、印刷字和多个相邻图形的分离仍不稳定；
- Apple 网页版本使用 CPU/WASM，不等于使用 Apple Neural Engine；后者需要原生 Core ML 应用；
- 物理 iPhone/iPad 的速度和内存仍需在目标机型上验收。

### “图2”案例为什么能稳定得到 7/10/0/4

本地真实照片回归会在题干同时匹配多条明确构造语句时启用专用题干模板。它把 U-Net 点候选与题干给出的拓扑相匹配，期望得到：

- 7 个点 A–G；
- 10 条可见线段，包含 E–G；
- 0 个箭头；
- 4 个点在线段上的关系。

这里的 A–G 来自“题干拓扑匹配”，不是字母 OCR；10 条线也受到已知题干模板约束。因此该回归证明的是题干约束、端侧推理、人工校正和分享闭环可以工作，不能当作通用照片识别准确率。题干不匹配该模板时，系统会返回保守的通用候选并要求人工复核。

公开网页里的“打开图2案例”不包含教材照片和训练权重，会直接载入一份人工核对过的自绘 7/10/0/4 结构，供用户体验编辑与分享流程；它不是一次模型预测。真实照片端侧回归必须由使用者在本地提供有权使用的图片与 ONNX 权重。

## 系统流程

~~~mermaid
flowchart LR
    A["题目照片 / 选区"] --> B["原始像素裁剪与旋转"]
    B --> C["GeometryUNet<br/>1×3×320×320"]
    C --> D["六类像素掩码"]
    D --> E["确定性图元与拓扑提取"]
    E --> F{"已知题干模板匹配？"}
    F -- 是 --> G["题干约束的候选匹配"]
    F -- 否 --> H["通用保守候选"]
    G --> I["人工校正工作台"]
    H --> I
    I --> J["原图 + 结构说明 + geometry-diagram/v5"]
~~~

模型负责感知，规则负责把像素变成可解释关系，用户负责最终事实确认。更多细节见 [架构说明](docs/ARCHITECTURE.md) 和 [模型卡](MODEL_CARD.md)。

## 真实测试结果

正式结果保存在 [ml/evaluation/pgdp5k_test_epoch5.json](ml/evaluation/pgdp5k_test_epoch5.json)。评估集为 PGDP5K 的 1,000 张测试图，输入统一为 320×320。

| 指标 | 结果 |
| --- | ---: |
| 像素准确率 | 0.983167 |
| 测试掩码中实际出现的 4 类平均 IoU | 0.652959 |
| 实线掩码 Precision / Recall | 0.539047 / 0.928956 |
| 曲线掩码 Precision / Recall | 0.560995 / 0.904029 |
| 点掩码 Precision / Recall | 0.626496 / 0.889827 |
| 点拓扑代理 Precision / Recall | 0.895616 / 0.975170 |
| 连接边拓扑代理 Precision / Recall | 0.749347 / 0.933475 |
| 完全一致的掩码代理图比例 | 0.301 |

必须同时阅读以下边界：

1. 点/边拓扑指标是从“预测掩码”和“真值掩码”分别提图后比较得到的可复现代理指标，不是直接对 PGDP5K 原始关系标注评分。
2. 匹配参数是 10 像素点容差与 0.58 线支持阈值。
3. 转换后的测试掩码没有虚线和文字像素，因此本次测试不能验证虚线分类或字母 OCR。
4. 像素准确率受到大面积背景影响，不能单独代表图结构质量。
5. 完全一致代理图比例只有 30.1%，说明通用自动输出仍必须经过人工确认。

## 快速开始

### Web 校正工作台

要求 Node.js ≥ 22.13.0。

~~~bash
npm install
npm run dev
~~~

浏览器打开 http://127.0.0.1:3000/。

工作台、人工校正、示例图和结构化导出可直接运行；没有模型文件时会明确降级为低置信度传统视觉候选。要启用 U-Net 端侧识别，请先从你有权使用的 checkpoint 导出 ONNX 到 `public/models/geometry_unet_pgdp5k_epoch5_320.onnx`。该文件被 `.gitignore` 排除，不会意外提交。首次端侧识别会从当前站点加载模型，并从固定版本 CDN 加载当前平台需要的 ONNX Runtime 文件；所选题图不会发送到 CDN 或推理 API。

常用检查：

~~~bash
npm run lint
npm test
npm run build
npm run build:static
~~~

tests/semantics-smoke.ts 覆盖箭头、线内位置、编辑后关系保留、图2必画线和 v5 分享语义；tests/vision-smoke.ts 覆盖传统视觉候选；tests/rendered-html.test.mjs 覆盖页面与 PWA 外壳。以当前 package.json 中接入的测试脚本为准。

### 数据转换

数据不随仓库分发。请先按 [data/README.md](data/README.md) 核对来源与使用条款。

~~~bash
python ml/prepare_pgdp5k.py \
  --root /path/to/PGDP5K \
  --out data/pgdp5k/processed

python ml/synthetic_dataset.py \
  --out ml/data/synthetic_v2 \
  --count 100
~~~

### GPU 训练

已验证训练在 CUDA 混合精度下使用 batch 16。训练代码会在报告的专用显存不超过 9 GiB 时拒绝更大的 batch，以避免 Windows/WDDM 显存压力导致严重卡顿。

~~~bash
python ml/train.py \
  --data ml/data/synthetic_v2 \
  --real-data data/pgdp5k/processed \
  --real-repeat 1 \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 2 \
  --amp \
  --require-cuda \
  --cuda-memory-fraction 0.75 \
  --out ml/checkpoints/geometry_unet_pgdp5k_gpu.pt
~~~

继续训练时必须显式传入原 checkpoint：

~~~bash
python ml/train.py \
  --data ml/data/synthetic_v2 \
  --real-data data/pgdp5k/processed \
  --real-repeat 1 \
  --epochs 6 \
  --batch-size 16 \
  --num-workers 2 \
  --amp \
  --require-cuda \
  --cuda-memory-fraction 0.75 \
  --resume ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --out ml/checkpoints/geometry_unet_pgdp5k_gpu.pt
~~~

### 正式评估

~~~bash
python ml/evaluate.py \
  --data data/pgdp5k/processed \
  --checkpoint ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --split test \
  --batch-size 16 \
  --num-workers 2 \
  --out ml/evaluation/pgdp5k_test_epoch5.json
~~~

### 单图推理与 ONNX 导出

~~~bash
python ml/infer.py input.png \
  --checkpoint ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --out-mask ml/outputs/input.mask.png \
  --out-json ml/outputs/input.graph.json

python ml/export_onnx.py \
  --checkpoint ml/checkpoints/geometry_unet_pgdp5k_gpu.pt \
  --out public/models/geometry_unet_pgdp5k_epoch5_320.onnx
~~~

Python 依赖见 ml/requirements.txt。ONNX 导出还需要 onnx 包。GPU 环境必须安装与实际显卡架构匹配的 PyTorch/CUDA 版本，不要把某台机器的 CUDA wheel 组合盲目复制到所有设备。

## 跨平台运行

| 平台 | 当前路径 | 已验证边界 |
| --- | --- | --- |
| Windows Edge/Chrome | 优先 WebGPU，失败时 CPU/WASM | 两条路径均完成图2回归；记录的单次时延分别为 1,190 ms 与 985 ms |
| iPhone/iPad Safari | 单线程 CPU/WASM | 代码路径已完成；物理设备性能仍需逐机型验收；不使用 Neural Engine |
| Android Chromium | 支持时尝试 WebGPU，否则 CPU/WASM | 条件式支持，仓库没有给出实体 Android 验收结果 |
| 本地 CUDA 服务 | batch 1、混合精度，仅作桌面诊断 | 生产 Web UI 不依赖该服务 |

## 分享给多模态模型

桌面剪贴板无法可靠地一次粘贴图片和文本，因此采用明确的两步流程：

1. 复制并粘贴干净的原题图片；
2. 在同一会话中复制并粘贴结构说明与 geometry-diagram/v5 JSON。

Apple 设备可通过系统分享表单发送原始 PNG 与结构附件。分享说明要求接收方：

- 先从原图读取印刷题干；
- 再用自身多模态能力观察图形；
- 最后用点、线、共线顺序和 attachments 做辅助核对；
- 若结构说明与题干冲突，逐项指出并以原图题干为最高依据；
- 不把近似坐标推断成中点、等长、垂直或精确比例。

## 目录

| 路径 | 作用 |
| --- | --- |
| app/ | Web 工作台、端侧推理、通用视觉规则、题干约束与结构导出 |
| static/ | GitHub Pages 静态入口 |
| public/models/ | 本地 ONNX 放置目录；权重默认不公开提交 |
| ml/ | 数据转换、U-Net、训练、推理、评估和导出 |
| ml/evaluation/ | 可提交的正式评估报告 |
| tests/ | 页面、视觉、语义与图2回归 |
| data/README.md | 数据来源与许可边界；不含原始数据 |
| docs/ | 架构和简历表达 |
| docs/DEVELOPMENT_LOG.md | 完整工程日志、硬件证据与历史决策 |

## 已知限制

- 复杂全页照片若不先选择正确图形区域，文字和手写笔迹会制造大量假点、假线；
- 320×320 统一输入会丢失很小的字母与细标记；
- 当前分割训练目标与最终“关系图正确”目标并不完全一致；
- 高边召回伴随较低边精确率，假连接抑制仍是主要问题；
- 箭头依赖后处理和人工确认，没有独立训练的箭头类别；
- 通用照片下点名仍是临时 ID；
- 当前题干约束只覆盖已实现的明确模板，不会自动泛化为任意几何语言解析器；
- 原始数据、示例照片和衍生权重的再分发/商业使用条款需要分别核对，见 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 路线图

- [ ] 用原始关系标注或人工审计集替代仅掩码派生的拓扑代理评估；
- [ ] 分离印刷几何线、文字、手写笔迹、角标和箭头；
- [ ] 增加点字母分类与置信度，而不是把 P1/P2 当作识别结果；
- [ ] 训练关系感知或任务引导损失，优先降低假连接；
- [ ] 把题干约束从单一模板扩展为可审计的通用关系解析；
- [ ] 建立更多真实手机照片回归集并完成 iPhone/iPad 实机验收；
- [ ] 明确数据、示例图片和模型权重的公开许可；
- [ ] 如确有性能需求，再评估原生 Core ML 与 Apple Neural Engine。

## 适合怎样使用这个项目

它更适合作为“端侧视觉 + 确定性规则 + 人工确认 + 可追踪输出”的工程样例，而不是宣称已经解决几何识别。该模式也能迁移到光谱、缺陷检测、多模态传感和工业过程数据：模型先生成候选，领域规则约束结构，人类确认不确定项，最终输出可审计数据。

面试和简历表达见 [docs/RESUME_PROJECT.md](docs/RESUME_PROJECT.md)。

## 许可与第三方材料

本仓库中的依赖、数据来源、示例图片和训练权重不是同一种许可对象。当前尚未附加根开源许可证，因此公开可见不等于获得复制、再分发或商业使用授权。进一步发布或商业化前，请阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，并分别核对各上游条款。
