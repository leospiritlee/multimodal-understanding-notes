# 第2章 切图与视觉编码

**一句话**：`patch` 与 `image encoder` 决定一张图变成多少 `image token`，也决定后续所有关口的上界。

**为什么需要它**：一张十二兆像素的手机照片和一张缩略图，进同一个模型会变成数量级不同的 token。综合报告口径给出量级：固定分辨率的 `LLaVA-1.5` 每图约五百七十六个 `image token`，`Qwen2.5-VL` 处理十二兆像素照片可达一万五千余个，checkpoint 默认上限一万六千余个。若不在这一关建立「像素即成本」的直觉，后面所有容量规划都会按纯文本口径算错。

**核心结构**：`ViT` 把图像切成固定大小的 `patch`，每个 `patch` 当作一个 token；`CLIP` 用 `contrastive learning` 在海量的 `image-text pair` 上训练 `image encoder`，让图文进入同一 `embedding space`；`ALIGN` 用带噪图文对扩大数据规模；`SigLIP` 把 softmax 对比换成 `sigmoid loss`，不再依赖全局 batch；`SigLIP 2` 增加多语言与稠密特征并支持原生宽高比；`DINOv2` 走自监督路线，局部特征强、适合 `vision-centric` 的密集任务；`SAM` 用掩码监督提供边界感知，常作第二编码器。

**关键设计**：综合报告口径强调，训练方法比编码器规模更重要——四亿参数的 `SigLIP 2` 在多数基准上超过五十九亿参数的 `InternViT-6B`。编码器对 `document understanding`、空间推理这类视觉中心任务影响显著，对语言推理主导的任务影响较小。更关键的是，视觉塔的 `frozen` 状态是默认：换编码器意味着特征分布变化，对齐层乃至部分预训练要重做，收益可能小于换分辨率策略。

**拆解**：
### 1. patch 大小决定 token 数量级
`ViT` 的 `patch size` 与后续合并策略直接决定序列长度。`Qwen2-VL` 用 `patch14` 切图后每二乘二合并，一个 token 代表二十八乘二十八像素，token 数约等于高除以二十八乘宽除以二十八。`image token` 不是免费附件，它们和文字 token 一起占 `KV cache`。

### 2. 对比学习解决了什么
`CLIP` 的 `contrastive learning` 让图像和文本在同一个 `embedding space` 对齐，`zero-shot` 迁移因此可用。`SigLIP` 的 `sigmoid loss` 去掉了对超大 batch 的依赖，小 batch 也稳。这条路线确立的对齐能力，是后面 `visual instruction tuning` 能起步的前提。

### 3. 自监督与掩码监督的分工
`DINOv2` 的自监督特征局部性强，适合分割、检测与 `grounding`；`SAM` 的掩码监督边界感知极强，适合确定区域落点。综合报告口径说，这两类编码器语义与文本对齐偏弱，常作为第二编码器与 `CLIP` 或 `SigLIP` 混用，而不是替换主视觉塔。

### 4. 别轻易换塔
`image encoder` 的输出空间要经过对齐训练才能和语言模型对话。换成单看更强的塔，也要重训 `projector`，净收益未必为正。务实做法是沿用底座自带的视觉塔，把精力投到数据与任务头。

**今天怎么做**：先明确输入图像的尺寸分布，记下第 50 与第 99 百分位；对底座自带的 `image encoder` 与 `patch size` 做一次 token 数实测，而不是照文档默认；只有域偏移严重且域内标注充足时，才考虑解冻或换塔。

<div class="chapend" markdown="1">

**容易误解的地方**：把编码器参数量当成能力指标是错的；`frozen` 也不是偷懒，而是保住大规模图文对学到的表示。视觉侧的瓶颈常在分辨率与 token 传递方式，不在参数量。

**记住这一句**：切图决定上界——小字在 patch 阶段丢了，后面再强的模型也读不回来。

</div>
