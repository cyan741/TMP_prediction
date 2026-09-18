# ATM-TCR 接入及执行记录

ATM-TCR 输入为肽和单条 TCR β CDR3。两者共享氨基酸 embedding，分别经过
5头自注意力，展平拼接后经2048→1024→1分类器输出 sigmoid 分数。
模型不使用 α 链或 MHC。原始网络、词表、中间 padding 和残基映射均已复用。
详细逻辑：`../../../Agentic-TPH-training-code/ATM_TCR_BASELINE.md`。

## 官方模型直接预测：已完成

- 权重：官方 Lee-CBG/ATM-TCR 的 `models/original.ckpt`，来源及哈希见 `checkpoint_provenance.json`。
- 输入：`/root/TMP_prediction/raw_test_sets/immrep2025_dataset.tsv`，8938行、20种肽。
- 结果：`../../predictions/ATM-TCR/immrep25/predictions.csv`。
- AUROC：0.5180966222；AUPRC：0.1045781161。
- 固定阈值：0.5。全部原始字段及行顺序已验证保留，预测 worker 不接收标签。
- 指标和核对记录：同目录 `metrics.json`、`verification.json`。

## 固定 workspace 训练后预测：已启动，训练持续运行

- workspace：`../../../Agentic-TPH-training-code/workspaces/hitph0630_unseen_seed42/hitph`。
- 保留106918条训练正样本；每 epoch 动态1:1负采样，使用投影到β链的已知配对排除表。
- 验证11691行，测试11834行；训练/验证/测试肽集合互不重叠。
- 参数：seed=42，negative_seed=42，batch=32，Adam lr=0.001，最多100 epochs，patience=4。
- 按验证 AUROC 保存最佳模型，按验证 MCC 冻结阈值；训练完成后自动预测固定测试集并计算距离分层指标。
- 首 epoch 已完成：训练 loss=0.3992837894，验证 AUROC=0.7016494512。这是启动验证记录，不是最终测试指标。
- 输出目录：`../../predictions/ATM-TCR/hitph0630_unseen_seed42/`，最终输出 `predictions.csv` 和 `metrics.json`。
- 训练日志：该目录 `train.log`；模型及历史记录在 `model/`。
- 实时整体状态：本目录 `run_status.json`、`execution.log`。
- β链投影后的验证/测试标签冲突组数分别15/2，原始记录均保留，详情见输出 `plan.json`。

## 环境和验证

- 模型独立环境：`../../envs/atm-tcr`，Python3.8.20、PyTorch1.10.0+cu113、torchtext0.11.0。
- NumPy1.20.1、SciPy1.6.1、scikit-learn0.24.1等使用原仓库固定版本；完整版本见 `runtime-freeze.txt`。
- 宿主 CLI 环境：`../../envs/baseline-host`，版本见 `host-freeze.txt`。
- GPU：RTX3080Ti；CUDA11.3可用，实际 GPU 矩阵运算通过，见 `gpu_probe.txt`。
- 官方 CUDA wheel 哈希已通过，见 `torch_artifact_verified.json`。
- 10项现有 baseline 回归测试、4项 ATM 接入测试通过；原版 GPU 小数据训练→预测全链路通过，见 `integration_smoke_result.json`。
- 环境重建：`install.sh`；运行入口：`run.sh immrep25` 或 `run.sh train`。这些脚本会拒绝覆盖已完成的预测，当前真实训练无需再次启动。
