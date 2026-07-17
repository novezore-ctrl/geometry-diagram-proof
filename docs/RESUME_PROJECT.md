# GeoStruct 简历与面试表达

## 推荐定位

把 GeoStruct 定位为“人在回路的端侧视觉结构化系统”，不要写成“通用几何识别已经解决”或“AI 自动解题平台”。

它最有价值的工程链条是：

> 图像感知 → 确定性结构恢复 → 不确定性表达 → 人工校正 → 可追踪 JSON → 跨设备端侧部署

这条链条能迁移到缺陷检测、光谱/传感数据审核、RGB–HSI 重建质量复核、工业过程异常标注和数字孪生边缘输入。

## 中文简历版

### 三条版

**GeoStruct：人在回路的几何图结构化与端侧推理系统｜个人工程项目**

- 使用 PGDP5K 的 3,500 张训练图和 500 张验证图训练轻量六分类 U-Net，构建“图元分割—确定性拓扑提取—人工纠错—geometry-diagram/v5 导出”的端到端流程，并把箭头、点在线段上与等分点确认建模为显式关系。
- 在 1,000 张测试图上取得点拓扑代理 Precision/Recall 89.6%/97.5%、连接边拓扑代理 Precision/Recall 74.9%/93.3%；完整掩码代理图一致率为 30.1%，据此诚实定位文字笔迹、箭头和假连接为主要失败来源。
- 将 2.2 MB ONNX 模型接入浏览器：Windows 优先 WebGPU，iPhone/iPad 使用 CPU/WASM；实现缩放旋转、屏幕尺寸固定覆盖层、关系校正与原图+结构化说明的多模态分享，并把真实漏线和坐标回归固化为测试。

### 一条压缩版

开发 GeoStruct 端侧几何图结构化原型：训练轻量 U-Net 并以确定性规则恢复点线拓扑，在 1,000 张 PGDP5K 测试图上获得点/边拓扑代理 P/R 89.6%/97.5% 与 74.9%/93.3%，完成 WebGPU/WASM 跨设备推理、人工纠错和可追踪 JSON 输出。

## English resume version

**GeoStruct — Human-in-the-loop geometry diagram parser**

- Trained a compact six-class U-Net on PGDP5K and built an end-to-end pipeline from primitive segmentation to deterministic topology extraction, human correction, and geometry-diagram/v5 export.
- Evaluated 1,000 test diagrams with mask-derived proxy point precision/recall of 89.6%/97.5% and edge precision/recall of 74.9%/93.3%; reported the 30.1% exact proxy-graph rate and used failure cases to prioritize false-edge suppression.
- Exported a 2.2 MB ONNX model for in-browser inference using WebGPU on compatible Windows browsers and CPU/WASM on iPhone/iPad, with zoom-invariant overlays, editable relations, uncertainty-aware attachments, and regression tests derived from real user failures.

## 30 秒项目介绍

我做的不是一个自动解题器，而是把几何题照片变成可人工确认的结构图。前端先在设备本地运行一个轻量 U-Net，分出线、曲线和点，再用确定性规则恢复连接关系；机器不确定的点名、箭头和线内位置由用户修正，最后输出带来源和不确定性的 JSON，并与原图一起交给多模态模型核对。这个项目让我完整经历了 GPU 训练、代理指标设计、端侧部署、坐标系统回归和真实用户失败驱动的测试闭环。

## 面试时必须主动说明的边界

### 1. 为什么像素准确率 98.3%，完整图只有 30.1%？

背景像素占比很高，像素准确率容易显得很好；结构任务中一处假点就可能组合出多条假边。完整图率更严格，也更接近产品痛点。项目因此同时报告线/点 P/R、拓扑代理 P/R 和完整图率，而不是只展示像素准确率。

### 2. 拓扑指标为什么叫“代理”？

评估代码从预测掩码和真值掩码分别提取图，再按 10 像素点容差和 0.58 线支持阈值比较。它可复现，但没有直接使用 PGDP5K 原始关系标注，所以不能称为最终关系识别准确率。

### 3. 图2为什么表现好？

图2有专用、严格触发的题干模板。U-Net 提供点候选，模板给出应有的 A–G、线段和共线关系，算法再做候选分配。因此 7 点、10 线、0 箭头、4 个线内位置是“模型+题干约束+固定回归案例”的验收，不是通用照片成绩，点名也不是字母 OCR。

### 4. 为什么需要人工校正？

拍照中的印刷字、手写笔迹和角标会进入点/线类别。把候选结果直接交给大模型会让视觉错误升级成题目条件。人工校正层把来源、置信度、近似位置和用户确认分开，能阻止系统静默编造几何事实。

### 5. 为什么没有直接使用大模型做全部识别？

本地小模型与规则可以在设备端运行、结果可解释、错误可编辑，也避免把题图上传到推理服务。多模态大模型放在上层负责结合原题图片和题干理解，而不是替代底层可追踪结构。

### 6. 项目中最重要的工程问题是什么？

加入缩放后曾出现覆盖层被压缩到错误区域。继续调阈值无效，最终定位到源图、旋转、选区、模型和 CSS 显示坐标混用。修复后把坐标变换拆开，并用 100%/200% 缩放一致性、选区归一化和真实图回归固定下来。

## 可迁移能力

| GeoStruct 中的模块 | 可迁移到的研究/工业任务 |
| --- | --- |
| U-Net 图元分割 | 食品表面缺陷、污染区域、组织结构分割 |
| 候选→规则→人工确认 | 小样本光谱异常审核、传感器事件标注 |
| 代理指标与失败审计 | RGB–HSI 重建的任务相关评价、数据泄漏排查 |
| WebGPU/WASM 端侧推理 | 无服务器演示、现场检测、隐私敏感数据 |
| 带来源和不确定性的 JSON | 多模态融合、数字孪生输入、可追踪决策 |
| 真实失败转回归测试 | 科研代码工程化和长期复现 |

## 不要写进简历的说法

- “实现任意几何图 98% 准确识别”；
- “已经完成 A/B/C/D OCR”；
- “图2 证明通用识别达到 100%”；
- “iPhone 使用 Apple Neural Engine”；
- “自动理解并证明几何题”；
- “PGDP5K 所有类别都完成验证”；
- “模型可直接商业化”，除非数据、示例图片与权重许可已经单独确认。

## 项目价值排序

对求职和后续研究，建议按以下顺序介绍：

1. 真实失败驱动的系统迭代；
2. 端到端工程闭环与跨设备部署；
3. 评价边界和不确定性意识；
4. 轻量模型训练与结构后处理；
5. 几何题这一具体场景。

这样能让项目从“小众几何工具”转化为一个可复用的可靠 AI 系统案例。
