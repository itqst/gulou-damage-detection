# run9 最终结果索引

## 模型目录

- 增冲单案例模型：`run9/zengchong/model/train/weights/best.pt`
- 朝利单案例模型：`run9/chaoli/model/train/weights/best.pt`
- 从江单案例模型：`run9/congjiang/model/train/weights/best.pt`
- 三鼓楼通用模型：`run9/universal/model/train/weights/best.pt`

## 文档

- `zengchong\docs\增冲鼓楼破损检测单案例实验报告.docx`
- `chaoli\docs\朝利鼓楼破损检测单案例实验报告.docx`
- `congjiang\docs\从江鼓楼破损检测单案例实验报告.docx`
- `universal\docs\三鼓楼通用破损检测模型实验报告.docx`
- `fusion\docs\三鼓楼实例融合实验报告.docx`

## 数据集统计

| 数据集 | 图像数 | 正样本 | 负样本 | 标注框 | 正样本比例 |
|---|---:|---:|---:|---:|---:|
| 增冲鼓楼 | 236 | 170 | 66 | 735 | 72.03% |
| 朝利鼓楼 | 228 | 164 | 64 | 376 | 71.93% |
| 从江鼓楼 | 268 | 193 | 75 | 769 | 72.01% |
| 三鼓楼通用数据集 | 732 | 527 | 205 | 1880 | 71.99% |

## 关键测试指标

| 模型/测试集 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| zengchong_single_on_zengchong_test | 0.867 | 0.873 | 0.890 | 0.394 |
| chaoli_single_on_chaoli_test | 0.750 | 0.468 | 0.545 | 0.261 |
| congjiang_single_on_congjiang_test | 0.719 | 0.595 | 0.612 | 0.186 |
| universal_on_universal_test | 0.855 | 0.691 | 0.755 | 0.317 |
| universal_on_zengchong_test | 0.924 | 0.880 | 0.887 | 0.433 |
| universal_on_chaoli_test | 0.802 | 0.656 | 0.702 | 0.311 |
| universal_on_congjiang_test | 0.815 | 0.527 | 0.618 | 0.213 |