# 00 · 项目上下文 〔本项目活记忆 · AI 维护〕

> **作用**:这是项目的"身份档案"。AI 接管项目时先读这里,了解项目目标、技术栈、目录、部署取值。
> **更新时机**:架构、技术栈、目录结构、端口、部署目录、重要约束变化时更新。

---

## 1. 项目是什么

- **项目名称**:`banksys`
- **一句话目标**:基于客户画像与营销活动数据,预测客户是否会购买银行定期存款产品。
- **使用者/受益者**:银行营销团队 — 用于精准筛选目标客户,降低营销成本,提升转化率。
- **核心功能**:
  - 数据探索与预处理:加载、清洗、特征工程
  - 模型训练与评估:训练分类模型,输出 Accuracy 指标
  - 预测 API:提供单条/批量预测接口
  - 健康检查与部署:容器化部署,CI/CD 自动上线
- **输入/数据**:葡萄牙银行营销数据集(`data/train.csv` 约 2.8MB, `data/test.csv` 约 0.9MB)；字段说明见 `data/字段说明.md`；非敏感数据,公开教学用途,应纳入 Git。

## 2. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.11 | 机器学习生态最成熟,课程默认版本 |
| ML 框架 | scikit-learn | 二元分类任务首选,简单可解释 |
| 数据处理 | pandas / numpy | 表格数据标准工具 |
| Web/API 框架 | FastAPI | 轻量、高性能、自带 OpenAPI 文档 |
| 测试 | pytest + pytest-cov | 与 CI 集成,覆盖率报告 |
| 格式/静态检查 | ruff | 替代 flake8/isort/black,单一工具,速度快 |
| 打包/运行 | Docker | 统一运行环境,便于部署 |
| CI/CD | GitHub Actions | 通用、可视化、适合教学与团队协作 |

## 3. 目录地图

```text
banksys/
├── standards/                  # AI 项目记忆与通用规范
├── data/                       # 数据集(公开教学数据,纳入 Git)
│   ├── train.csv
│   ├── test.csv
│   └── 字段说明.md
├── src/                        # 源代码
│   ├── __init__.py
│   ├── preprocess.py           # 数据预处理
│   ├── train.py                # 模型训练与评估
│   └── api.py                  # FastAPI 预测服务
├── tests/                      # 测试
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_api.py
├── models/                     # 训练产物(不进 Git)
├── requirements.txt            # 生产运行依赖
├── requirements-dev.txt        # 本地/CI 检查依赖
├── Dockerfile                  # 容器构建
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
└── README.md
```

> 新增目录前先更新本节,避免项目越做越散。

## 4. 质量门槛

| 类型 | 本项目标准 |
|---|---|
| 格式检查 | `ruff format --check .` |
| 静态检查 | `ruff check .` |
| 单元测试 | `pytest` |
| 覆盖率 | >= 80% |
| 构建 | `docker build` 成功 |
| 业务/模型指标 | Accuracy >= 0.85(基线);最终目标以调优结果为准 |

## 5. 不变约束

- 密钥、密码、私钥、Token **绝不写进代码或文档**,只进 GitHub Secrets / 环境变量。
- 数据集(`data/*.csv`)为公开教学数据,**纳入 Git**。
- 模型产物(`models/*`)**不进 Git**,由训练脚本生成。
- `main` 分支受保护,日常开发必须走 feature 分支 + PR。
- CI 红灯不合并。

## 6. 部署/CI 占位符取值

> `guides/` 和 workflow 里的通用占位符,在本项目里的真实值只写这里。

| 占位符 | 本项目取值 | 说明 |
|---|---|---|
| `<APP>` | `banksys` | 应用名/镜像名/容器名 |
| `<DEPLOY_DIR>` | `/opt/banksys` | 服务器部署目录 |
| `<PORT>` | `8000` | 服务端口 |
| `<PYVER>` | `3.11` | Python 版本 |
| `<HEALTHCHECK>` | `/health` | 健康检查地址 |
| `<SSH_USER>` | `root` | 部署用户 |
| `<SSH_HOST>` | `<待填写>` | 服务器公网 IP 或域名 |
