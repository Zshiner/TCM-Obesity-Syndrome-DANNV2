# LLM 复现性披露（自动生成）

本文件对**每个 LLM (hard-label) 模型目录无条件生成**（独立于 SHAP，故
LLM 模型即使不跑 SHAP 也必有此说明）。当审稿人在 `final_test_metrics.csv`
看到本模型跨 5 个 seed 的 `seed_std=0.0` 时，其方法学解释如下：

> 仅适用于 LLM (hard-label) 模型；ML/DL 模型不涉及，可忽略本节。
- LLM 走"缓存=确定性输入→输出映射"协议：response 按 (provider / api_model_id
  / request_params / rendered messages / prompt_protocol_version) 的 SHA-256
  cache_key 缓存（compute_cache_key），去 seed 后**同输入命中同一缓存条目**。
  故 LLM 跨 seed 的指标方差恒为 0 是**预期行为、非 bug**——多 seed 不重采 LLM，
  仅重采框架级随机性（用户级 seed 由 V10 拒绝，多 seed 走 ExperimentConfig.seeds）。
- temperature = provider 官方默认（未在 param_space / fixed_params 显式指定时
  不下发该字段，由 provider 端取默认值）。
- 每样本**单次采样**（无 self-consistency / 多次投票），**未估计采样方差**；
  报告的是单条采样响应在确定性缓存下的复现，而非采样分布的期望/置信区间。
