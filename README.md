# WeaverRT — 轻量级 AI Agent 生产与进化基础设施

**零重框架依赖**（Python 3.13 标准库 + PyYAML + WSL2/Linux 内核隔离）手写实现的 AI Agent Infra 系统，完整打通 **`Build → Evaluate → Gate → Release → Run → Observe → Evolve`** 全生命周期。

---

## 一、架构与六大核心子系统

```
weaverrt/
├── domain.py                     # 领域模型：Deployment / Workspace / Session / Turn / Event / TraceSpan
├── kernel/                       # [M1] Harness Kernel（执行内核）
│   ├── model.py                  #   OpenAI 兼容 HTTP 适配器 + ModelRouter 多模型异构路由/降级 + Replay 桩
│   ├── tools.py                  #   版本化 ToolRegistry + Schema 校验 + 幂等执行 + Secret 输出脱敏
│   ├── skills.py                 #   SkillRegistry（SKILL.md 声明式技能包 + 渐进式按需披露）
│   ├── approval.py               #   Agent Identity / RBAC / HMAC 委托授权（防提权/深度限制/TTL） / 审批 Gate
│   ├── telemetry.py              #   层次化 Span Tracer（session -> turn -> context/llm/tool/verifier）
│   ├── workflow.py               #   Declarative JSON DAG Workflow（并行扇出/条件路由/节点级断点续跑）
│   └── architectures.py          #   同一内核可插拔运行 ReAct / Workflow / Planner-Executor / Multi-Agent(Solver+Verifier)
├── runtime/                      # [M2] Managed Runtime（长任务可靠执行）
│   ├── store.py                  #   SQLite WAL 事件溯源 + 乐观锁 Lease 租约接管 + Exactly-Once 幂等缓存
│   ├── worker.py                 #   多 Worker 接管、Token/成本预算熔断、审批挂起与恢复
│   └── supervisor.py             #   ElasticWorkerPool（弹性扩缩容 + 后台心跳续约 + 孤儿会话自动故障收割）
├── context/                      # [M3] Context Engineering & Layered Memory
│   ├── memory.py                 #   四层记忆（Working / Episodic 自动沉淀 / User Profile 个性化 / Semantic FTS5 KB）
│   └── engine.py                 #   硬 Token 预算、大工具输出外部化 Artifact、防断对压缩、Grounding 假引用核验器
├── mcp/ & sandbox/               # [M4] MCP 协议与真 Linux Namespace 安全沙箱
│   ├── mcp/server_demo.py        #   标准 MCP JSON-RPC 2.0 over stdio 服务端（动态发现 + 路径越界防护）
│   ├── mcp/client.py             #   MCP 客户端与 ToolRegistry 治理桥接
│   └── sandbox/linux_sandbox.py  #   Linux unshare(user+pid+net) + RLIMIT(CPU/AS/FSIZE) + env -i + PEP 578 审计钩子
├── control_plane/                # [M5] Agent Control Plane（控制面）
│   ├── registry.py               #   YAML Agent Artifact、Gate 强校验、Canary 灰度流量路由、一键回滚、HTTP BYO Agent 纳管
│   ├── server.py                 #   控制面 REST API（Artifacts / Sessions / Traces / Approvals / Evals / BadCases）
│   └── dashboard.html            #   实时可观测与人工审批 Web 控制台
└── eval/                         # [M6] Evaluation & Continuous Evolution（评测与进化闭环）
    ├── engine.py                 #   多维评估器（确定性/轨迹/安全/LLM-as-a-Judge）、版本回归 Diff、Gate 阻断、Bad Case 回流
    └── sampler.py                #   OnlineSampler（线上按比例 + 异常/假引用触发抽样，自动回流 Bad Case 回归集）
```

---

## 二、一键运行全部硬核验收测试

```powershell
cd C:\Users\link\.gemini\antigravity\scratch\weaverrt

# 1. 运行全部 M1-M6 自动化验收测试（含真实 OS 强杀混沌测试 + 6 项 Linux 沙箱安全攻防 + 20 用例回归 Gate）
python -m unittest discover -s tests -v

# 2. 运行端到端生产闭环 Demo（沙箱攻防 + v1 拦截 + Bad Case 回流 + v2 放行发布）
python examples/demo_full_lifecycle.py

# 3. 启动实时 Control Plane Web 控制台（浏览器打开 http://127.0.0.1:8787）
python examples/demo_full_lifecycle.py --serve
```

### 各模块测试覆盖一览
- `tests/test_m1_harness.py`：3 种执行架构切换、高危审批挂起跨 Worker 恢复、Secret 脱敏、预算熔断、OpenAI Wire 格式校验
- `tests/test_m2_chaos_recovery.py`：**真实子进程 OS 强杀（`kill`）混沌测试**——在外部账本写入 + 幂等表落盘之后、Turn 结束前强杀 Worker A，验证防脑裂租约、Worker B 接管恢复，以及外部账本**精确仅执行 1 次（Exactly-Once）**
- `tests/test_m3_context_memory.py`：中英双语 FTS5 Grounded RAG、100 KB 超大工具输出外部化 + 30 轮对话严格控窗（Token 降幅 > 90%）、Agent 按行回读 Artifact
- `tests/test_m4_mcp_and_sandbox.py`：真实 MCP JSON-RPC 2.0 动态发现调用 + **6 项真实安全攻击**（路径穿越、CPU 死循环、内存炸弹、外网外联、Fork 炸弹、环境变量窃取 & RBAC 越权）100% 拦截与审计
- `tests/test_m5_m6_control_plane_and_eval.py`：YAML Artifact 注册 → 20 用例基准评测 → `v1` 被 Gate 拦截（HTTP 409）→ 8 条 Bad Case 自动归因回流 → `v2` 100% 通过且 Token 下降 → HTTP API 发布 → 一键回滚 + BYO Agent 纳管

---

## 三、接入真实大模型（DeepSeek / 通义千问 / OpenAI / Ollama）

`OpenAICompatibleAdapter` 使用原生 HTTP 协议，无需安装第三方 SDK：

```python
from weaverrt.kernel.model import OpenAICompatibleAdapter

# 示例 1：DeepSeek
model = OpenAICompatibleAdapter(
    model="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
)

# 示例 2：本地 Ollama
model = OpenAICompatibleAdapter(
    model="qwen2.5:7b",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
```
