# PROGRESS · banksys 〔本项目活记忆 · 状态机〕

> **作用**:这是项目的"存档点"。任意 AI、任意重启会话,读它即可知道当前做到哪、下一步做什么、踩过什么坑。
> **更新时机**:每完成一个有意义步骤、每次会话结束前。
> **格式要求**:时间倒序,最新在上;短、准、可接力。

---

## 当前状态 (最后更新: 2026-06-07 · by AI)

- **阶段**:`第③步 — US-1~4 全部完成,等待确认后进入第④步(本地 CI 自检)`
- **上一步完成**:
  - ✅ US-1:工程化初始化(requirements, Dockerfile, CI/CD workflows)
  - ✅ US-2:数据预处理 — `src/preprocess.py`
  - ✅ US-3:模型训练 — `src/train.py`(LR + RF,RF Accuracy 0.8818)
  - ✅ US-4:预测 API — `src/api.py`(FastAPI, /health /predict /predict/batch)
  - ✅ 本地自检:59/59 测试 | ruff format ✅ | ruff lint ✅ | 覆盖率 97%
- **下一步 (TODO 第一条)**:用户确认 → push 分支 → 第⑤步创建 PR → CI → 人工合并 → CD
- **阻塞项**:无

---

## 待办清单 (TODO,按优先级)

- [x] 读取 standards/README.md 及 00~06 全部规范文件
- [x] 填写 `00-project-context.md`(项目身份、技术栈、目录地图、质量门槛)
- [x] 填写 `01-requirements.md`(US-1 到 US-5 用户故事与验收标准)
- [x] **用户确认** 00/01/PROGRESS 内容
- [x] 第①步 — 建仓:用 `gh` 创建 GitHub 仓库 `yhyllm/banksys` + 初始提交 + push ✅
- [x] 第①步 — 配 Secrets:SSH_PRIVATE_KEY / SSH_HOST / SSH_USER 已配置
- [x] 第②步:开第一条 feature 分支(`feature/1-init-engineering`)
- [x] US-1:初始化工程结构(`requirements.txt`, `src/`, `tests/`, `Dockerfile`, CI/CD workflows)
- [x] US-2:数据探索与预处理(`src/preprocess.py` + `tests/test_preprocess.py`)
- [x] US-3:模型训练与评估(`src/train.py` + `tests/test_train.py`)
- [x] US-4:预测 API 服务(`src/api.py` + `tests/test_api.py`)
- [ ] US-5:容器化与 CD 部署
- [ ] 本地 CI 自检通过(ruff + pytest + 覆盖率 >= 80%)
- [ ] PR → CI → 人工合并 → CD 部署 → 健康检查

---

## 关键决策记录 (ADR)

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-06-07 | 技术栈选型:Python 3.11 + scikit-learn + FastAPI | 课程常用版本;sklearn 适合入门二元分类;FastAPI 轻量高效,自带文档 |
| 2026-06-07 | 模型指标门槛:Accuracy >= 0.85 | 基线值,后续可按调优结果调整 |
| 2026-06-07 | 数据集纳入 Git | 公开教学数据,体积可控(合计 ~3.7MB),方便 CI runner 直接使用 |
| 2026-06-07 | 模型产物不进 Git | `models/` 目录 gitignore,由训练脚本在本地/CI 中生成 |
| 2026-06-07 | 选 Random Forest 为最佳模型(Accuracy 0.8818) | LR 0.7944 < 0.85 门槛;RF 虽 Recall 低(0.18)但 Accuracy 满足要求;评价标准为 Accuracy |

---

## 已知坑 (GOTCHAS)

- **PyYAML `on` 布尔陷阱**:PyYAML 1.1 将 `on` 解析为布尔 `True`,而非字符串。GitHub Actions 用 YAML 1.2 不受影响,但本地测试用 `yaml.safe_load` 读取 workflow YAML 时会失败;解决:测试用字符串匹配,不依赖 PyYAML 解析 `on` 键。

---

## 里程碑 (DONE)

- [x] 2026-06-07:standards 规范全部读取完毕
- [x] 2026-06-07:三个「项目活记忆」文件初始化完成,用户确认通过
- [x] 2026-06-07:第①步建仓 — GitHub 仓库 `yhyllm/banksys` 创建并推送成功
- [x] 2026-06-07:US-1 工程化初始化完成 — 16/16 测试全过,本地门禁全绿
- [x] 2026-06-07:US-2 数据预处理完成 — 30/30 测试,覆盖率 96%
- [x] 2026-06-07:US-3 模型训练完成 — 41/41 测试,覆盖率 97%,RF Accuracy 0.88
- [x] 2026-06-07:US-4 预测 API 完成 — 59/59 测试,覆盖率 97%18
