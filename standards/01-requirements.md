# 01 · 需求 / 活 PRD 〔本项目活记忆 · AI 维护〕

> **作用**:这是本项目唯一的需求文档。所有新功能、缺陷、技术债都追加到这里,不要另起多个 PRD 文件。
> **更新时机**:每次有新需求、需求变更、验收标准变化时更新。

---

## 1. 需求来源

| 类型 | 来源 | 进入方式 |
|---|---|---|
| 功能需求 Feature | 课程任务 / 模拟业务方 | 写成用户故事 |
| 缺陷 Bug | 测试 / 线上日志 / 用户反馈 | 写复现步骤和期望结果 |
| 技术债 Tech Debt | 开发 / Review / CI/CD 故障 | 写影响和修复目标 |

---

## 2. Issue 生命周期

| 阶段 | 状态 | 动作 |
|---|---|---|
| 提出 | Open | 写清场景、目标、验收标准 |
| 排期 | Backlog / Todo | 决定优先级和负责人 |
| 开发 | In Progress | 从 main 开 feature 分支 |
| 评审 | In Review | 提 PR,等待 CI 和 Review |
| 合并 | Done | PR 合并 main,自动关闭 Issue |
| 验收 | Verified | 按验收标准确认 |

**追踪规则**:分支名带 Issue 号,PR 描述写 `closes #<编号>`。

---

## 3. 用户故事模板

```text
### US-<编号> <一句话标题> · 状态: Backlog
作为 <角色>,
我想要 <能力>,
以便 <价值>。

验收标准:
- AC1: Given <前提>,When <动作>,Then <可验证结果>。
- AC2: <补充标准>

技术备注:
- <可选:约束、边界、风险>
```

---

## 4. 需求清单

### US-1 初始化项目工程化与 CI/CD · 状态: Backlog

作为 **项目开发者**,
我想要 项目具备基础工程结构、测试、CI 与 CD,
以便 后续每次开发都能自动检查并自动部署。

验收标准:
- AC1: 从 `main` 开 feature 分支完成初始化,不直接 push main。
- AC2: PR 触发 CI,至少包含格式检查、静态检查、单元测试、构建检查。
- AC3: CI 全绿后合并 main。
- AC4: 合并 main 自动触发 CD,部署后健康检查通过。
- AC5: 完成后更新 `standards/PROGRESS.md`。

---

### US-2 数据探索与预处理 · 状态: Backlog

作为 **数据科学家**,
我想要 加载原始数据并完成清洗与特征工程,
以便 为模型训练提供干净、规范的特征矩阵。

验收标准:
- AC1: Given `data/train.csv` 和 `data/test.csv`,When 运行预处理脚本,Then 输出包含所有特征列且无缺失值的特征矩阵 X 和标签 y。
- AC2: 分类变量(job, marital, education, contact, month, day_of_week, poutcome, default, housing, loan)正确编码为数值。
- AC3: 数值变量(age, duration, campaign, pdays, previous, emp_var_rate, cons_price_index, cons_conf_index, lending_rate3m, nr_employed)按需标准化或保留原值。
- AC4: 训练集和测试集经过相同预处理流程,避免数据泄露。
- AC5: 预处理逻辑有单元测试覆盖。

技术备注:
- 目标变量 `subscribe` 需从 yes/no 映射为 1/0。
- `pdays=999` 表示之前未联系,需特殊处理。
- `unknown` 值出现于多个分类字段,需统一处理策略。

---

### US-3 模型训练与评估 · 状态: Backlog

作为 **数据科学家**,
我想要 训练二元分类模型并评估 Accuracy,
以便 选出一个满足准确率门槛的模型用于预测。

验收标准:
- AC1: Given 预处理后的训练集,When 运行训练脚本,Then 输出训练好的模型文件到 `models/` 目录。
- AC2: 模型在测试集上的 Accuracy >= 0.85。
- AC3: 训练脚本输出分类报告(Accuracy, Precision, Recall, F1, Confusion Matrix)。
- AC4: 支持至少两种分类器(如 Logistic Regression, Random Forest),可选最佳模型。
- AC5: 训练逻辑有单元测试覆盖(含模型加载、预测输出格式)。

技术备注:
- 模型文件用 `joblib` 或 `pickle` 保存,不进 Git(已在 `.gitignore` 排除 `models/`)。
- 类别不平衡需关注:如果正负样本差距大,需考虑 class_weight 或采样策略。

---

### US-4 预测 API 服务 · 状态: Backlog

作为 **银行营销系统集成者**,
我想要 一个 RESTful API 接收客户数据并返回购买预测结果,
以便 将模型能力嵌入现有营销决策流程。

验收标准:
- AC1: Given 一条客户特征 JSON,When POST `/predict`,Then 返回 `{"subscribe": 0|1, "probability": float}` 格式的预测结果。
- AC2: Given 多条客户特征 JSON 数组,When POST `/predict/batch`,Then 返回批量预测结果。
- AC3: Given 服务运行中,When GET `/health`,Then 返回 `{"status": "ok"}` 及 HTTP 200。
- AC4: API 输入校验:缺少必要字段或类型错误时返回 422 及清晰错误信息。
- AC5: API 层有集成测试覆盖(使用 FastAPI TestClient)。

技术备注:
- 使用 FastAPI + uvicorn,自动生成 OpenAPI 文档。
- 模型在服务启动时加载一次,常驻内存。

---

### US-5 容器化部署与 CD · 状态: Backlog

作为 **运维人员**,
我想要 通过 Docker 容器化服务并经由 CI/CD 自动部署,
以便 每次合并 main 后自动更新线上服务。

验收标准:
- AC1: `Dockerfile` 可成功构建镜像,容器启动后 `/health` 可访问。
- AC2: `docker build` 在 CI 中执行成功。
- AC3: 合并 main 后 CD 自动触发,SSH 到服务器完成部署。
- AC4: 部署后健康检查通过,返回真实端口和访问地址。

---

## 5. 非功能需求

- **安全**:密钥只进 Secrets,不进 Git。
- **可维护**:一需求一小 PR,避免大爆炸式提交。
- **可测试**:核心逻辑必须有单元测试。
- **可部署**:部署后必须有健康检查或等价验证。
