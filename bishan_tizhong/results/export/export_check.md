# 导出核对

生成时间（北京时间）：2026-10-14T20:12:51+08:00
来源：`/root/autodl-tmp/bishan_exp/bishan_llm/bishan_tizhong/results`；导出：`/root/autodl-tmp/bishan_exp/bishan_llm/_export_v4/运行目录`；结果目录：`/root/autodl-tmp/bishan_exp/bishan_llm/_export_v4/运行目录/bishan_tizhong/results`

## 范围

- obesity_syndrome：knn、decision_tree、random_forest、extra_trees、dannv2_beta、realmlp、tabpfn_v3、tabicl、tabdpt、deepseek、glm、qwen（12 个学习器）
- 学习器 × 种子 共 60 组，全部齐全（不齐就不会走到这里）。

## 文件

- 复制 16015 个文件（191.3 MB），每个复制后重算 SHA-256 与源文件一致；不放 1635 个（36.8 MB）。模型文件：none；运行日志：不放。
- 逐文件清单：`export/file_index.csv`（path、learner、shard、bytes、sha256、exported、reason）。
- 不放的原因与条数：
  - 分片级 LaTeX 片段（框架副产品）：645
  - 运行日志 / 事件流（复现用不到）：450
  - 数据清单（由数据文件与划分记录派生）：180
  - 划分记录（同一种子的分片存的是同一份；每个种子一份放 shards/split_indices/）：180
  - 分片级的图（框架副产品；作图数据在 summary/main/curves/）：90
  - LaTeX 片段（框架副产品）：90
- 划分记录每个种子一份：`shards/split_indices/seed_<种子>/`。分片清单与分片配置不复制（写着原机器的绝对路径，`run` 时按定义重新生成）。
- 自定义代码目录 dannv2_beta：6 个文件
- 事后补算的部署开销 cost/：90 个文件
- 大模型的共享响应缓存 llm_cache/：7 个文件

## 在导出目录里重出汇总

- `freeze`：退出码 0（日志 `export/logs/freeze.log`）
- `check`：退出码 0（日志 `export/logs/check.log`）
- `status`：退出码 0（日志 `export/logs/status.log`）
- `report`：退出码 0（日志 `export/logs/report.log`）

## 与原目录同范围报告的比对（summary/main/*.csv 逐字节）

- main/compute_cost.csv：相同
- main/curves/calibration_bins_cv.csv：相同
- main/curves/calibration_bins_test.csv：相同
- main/curves/decision_curve_test.csv：相同
- main/cv.csv：相同
- main/cv_long.csv：相同
- main/data_description.csv：相同
- main/final_params.csv：相同
- main/framework_commits.csv：相同
- main/label_cooccurrence.csv：相同
- main/label_cooccurrence_matrix.csv：相同
- main/learner_params.csv：相同
- main/per_label_cv.csv：相同
- main/per_label_test.csv：相同
- main/test.csv：相同
- main/test_long.csv：相同
- main/test_paired_differences.csv：相同
- main/thresholds.csv：相同
- 跳过：learners.csv（学习器表来自各自的定义）。

## 结论

**通过**
