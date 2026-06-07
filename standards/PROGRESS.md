# PROGRESS · banksys 〔本项目活记忆 · 状态机〕

> **作用**:这是项目的"存档点"。任意 AI、任意重启会话,读它即可知道当前做到哪、下一步做什么、踩过什么坑。
> **更新时机**:每完成一个有意义步骤、每次会话结束前。
> **格式要求**:时间倒序,最新在上;短、准、可接力。

---

## 当前状态 (最后更新: 2026-06-07 · by AI)

- **阶段**:`初始化`
- **上一步完成**:读取 standards 全部文件,填写 `00-project-context.md` 与 `01-requirements.md`,初始化 `PROGRESS.md`。
- **下一步 (TODO 第一条)**:用户确认项目上下文与需求后,进入六步流程第①步 — 建仓 + 配 Secrets。
- **阻塞项**:等待用户确认三个「项目活记忆」文件的内容(特别是技术栈选择、模型指标门槛、部署服务器信息)。

---

## 待办清单 (TODO,按优先级)

- [x] 读取 standards/README.md 及 00~06 全部规范文件
- [x] 填写 `00-project-context.md`(项目身份、技术栈、目录地图、质量门槛)
- [x] 填写 `01-requirements.md`(US-1 到 US-5 用户故事与验收标准)
- [ ] **用户确认** 00/01/PROGRESS 内容
- [ ] 第①步:建仓 + 配 Secrets — 用 `gh` 创建 GitHub 仓库,提示配置 SSH_PRIVATE_KEY/SSH_HOST/SSH_USER
- [ ] 第②步:开第一条 feature 分支(`feature/1-init-engineering`)
- [ ] US-1:初始化工程结构(`requirements.txt`, `src/`, `tests/`, `Dockerfile`, CI/CD workflows)
- [ ] US-2:数据探索与预处理(`src/preprocess.py` + `tests/test_preprocess.py`)
- [ ] US-3:模型训练与评估(`src/train.py` + `tests/test_train.py`)
- [ ] US-4:预测 API 服务(`src/api.py` + `tests/test_api.py`)
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

---

## 已知坑 (GOTCHAS)

- 暂无(项目尚未进入开发阶段)

---

## 里程碑 (DONE)

- [x] 2026-06-07:standards 规范全部读取完毕
- [x] 2026-06-07:三个「项目活记忆」文件初始化完成,等待人工确认
