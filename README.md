# MyEvoSkill

MyEvoSkill is a doc-first scaffold for secure skill distillation and transfer
validation on scientific coding tasks.

The repository enforces:

- explicit `distill_train` and `transfer_val` splits
- hidden judging where a task passes only if all user metrics pass
- monotonic transfer validation: `success(no-skill) <= success(with-skill)`
- promotion of permanent skills only when they improve validation coverage
  without regressions
- isolation between public task bundles and hidden reference implementations

See `doc/` for the design specifications that govern the implementation.

## Task Registration

New raw tasks now enter MyEvoSkill through a fixed two-step contract-driven
registration flow instead of hand-editing `registry/tasks/*.json`.

Step 1: provide `tasks/<task_id>/evaluation/registration_input.json`, then
generate the draft contract with the mandatory Claude registration agent:

```bash
myevoskill-task-contract-draft --task-root ../tasks/conventional_ptychography --output-root .
```

Alternative module forms:
鍙€夋ā鍧楀舰寮忥細

```bash
python -m myevoskill.task_registration_draft --task-root ../tasks/conventional_ptychography --output-root .
PYTHONPATH=src python -m myevoskill.task_registration_draft --task-root ../tasks/conventional_ptychography --output-root .
```

If the first draft is structurally invalid, the harness feeds the validation
errors back to the registration agent and performs one automatic repair round.

Step 2: after the user confirms
`tasks/<task_id>/evaluation/registration_contract.json`, register it:

```bash
myevoskill-task-register --task-root ../tasks/conventional_ptychography --output-root .
```

Module form for confirmed registration after editable install:

```bash
python -m myevoskill.task_register --task-root ../tasks/conventional_ptychography --output-root .
```

Repo-local module form for confirmed registration without installation:

```bash
PYTHONPATH=src python -m myevoskill.task_register --task-root ../tasks/conventional_ptychography --output-root .
```

The draft step reads:

- `tasks/<task_id>/evaluation/registration_input.json`

The draft step writes:

- `tasks/<task_id>/evaluation/registration_contract.draft.json`
- `tasks/<task_id>/evaluation/contract_generation.notes.json`

The register step writes:

- `registry/tasks/<task_id>.json`
- `registry/tasks/<task_id>.notes.json`
- `tasks/<task_id>/evaluation/judge_adapter.py`

The confirmed source of truth is always
`tasks/<task_id>/evaluation/registration_contract.json`.

`registration_input.json` is the registration-stage semantic input. It tells
the Claude registration agent which files are task description, public input,
public metadata, evaluation logic, hidden references, and which metrics plus
thresholds define success. User input keeps the simple `operator` plus
`threshold` form; the generated contract normalizes those into
`judge_contract.metrics[*].pass_condition = { operator, threshold }`. It does
not inline README or script bodies.

Confirmed registration also builds or reuses one task runtime environment from
the task-local `requirements.txt`. The resulting manifest stores
`runtime_env.backend`, `runtime_env.env_hash`,
`runtime_env.python_executable`, and `runtime_env.ready`. If the requirements
file is missing or the environment build fails, registration fails and the task
is not marked live-ready.

See [doc/task_registration.md](doc/task_registration.md) for the full
registration flow and manual follow-up checklist.

## End-to-End Task Workflow / 任务使用手册

This is the recommended path for onboarding and running one task end to end.
下面是一条从注册到 live run 的推荐使用路径。

### 1. Prepare Registration Input / 准备注册输入

Create `tasks/<task_id>/evaluation/registration_input.json` before running the
draft step. This file is user-authored input, not generated output.
先准备 `tasks/<task_id>/evaluation/registration_input.json`。这是用户提供的输入文件，
不是 draft 命令生成的产物。

The file should declare:
该文件至少应声明：

- task description resources
- public input and public metadata resources
- evaluation logic resources
- optional hidden references and hidden metric config resources
- pass metrics with `operator` and `threshold`
- optional execution hints

### 2. Draft The Contract / 生成注册草稿

Run the mandatory Claude registration agent:
运行强制的 Claude 注册 agent：

```bash
myevoskill-task-contract-draft --task-root ../tasks/conventional_ptychography --output-root .
```

Artifacts written by the draft step:
draft 阶段会写出：

- `tasks/<task_id>/evaluation/registration_contract.draft.json`
- `tasks/<task_id>/evaluation/contract_generation.notes.json`

`contract_generation.notes.json` always contains `warnings`; when there is no
conflict it is written as `[]`.
`contract_generation.notes.json` 中始终会显式包含 `warnings` 字段；没有冲突时也会写成 `[]`。

### 3. Confirm The Contract / 人工确认正式合同

Review `registration_contract.draft.json`, then confirm it into:
检查 `registration_contract.draft.json`，确认后定稿为：

- `tasks/<task_id>/evaluation/registration_contract.json`

This confirmed contract is the frozen source of truth for formal registration,
judge generation, and live execution.
这个 confirmed contract 是正式注册、judge 生成和 live 执行的冻结真源。

### 4. Register The Task / 正式注册任务

Run formal registration:
执行正式注册：

```bash
myevoskill-task-register --task-root ../tasks/conventional_ptychography --output-root .
```

Alternative module forms:
可选模块形式：

```bash
python -m myevoskill.task_register --task-root ../tasks/conventional_ptychography --output-root .
PYTHONPATH=src python -m myevoskill.task_register --task-root ../tasks/conventional_ptychography --output-root .
```

Register does all of the following:
register 阶段会完成以下动作：

- validates the confirmed contract
- validates task-local file references used by resources and judge metrics
- builds or reuses the task runtime environment from `requirements.txt`
- writes `registry/tasks/<task_id>.json`
- writes `registry/tasks/<task_id>.notes.json`
- writes `tasks/<task_id>/evaluation/judge_adapter.py`

The task runtime environment is shared by:
这个 task runtime env 会被以下环节共用：

- agent-executed `python work/main.py`
- `python evaluation/self_eval.py`
- post-run audit and replay
- judge subprocess execution

### 5. Run Live / 运行真实 live test

Run the manifest-driven live runner:
运行 manifest 驱动的 live runner：

```bash
myevoskill-task-live --task-id conventional_ptychography --project-root .
```

Alternative module forms:
可选模块形式：

```bash
python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
PYTHONPATH=src python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
```

Live execution is gated by all of the following:
live 执行前必须满足以下条件：

- `tasks/<task_id>/evaluation/registration_contract.json` exists and is valid
- `manifest.judge_spec.ready == true`
- `manifest.runtime_env.ready == true`

If the cached runtime environment is missing locally, the live runner may
rebuild it from the registered `requirements.txt` and `env_hash`.
如果本地缓存的 runtime env 缺失，live runner 允许根据已登记的 `requirements.txt`
和 `env_hash` 进行重建。

### 6. Inspect Results / 查看结果

The live runner prints the log directory and workspace directory at the end of
the run. The main files to inspect are:
live runner 结束时会打印日志目录和工作目录。通常重点查看这些文件：

- `artifacts/logs/<task_id>/run-live-<timestamp>/judge_result.json`
- `artifacts/logs/<task_id>/run-live-<timestamp>/run_record.json`
- `artifacts/workspaces/<task_id>/run-live-<timestamp>/transcript.txt`

## Task Live Run

Registered tasks can be executed through one unified live-run entrypoint:

```bash
myevoskill-task-live --task-id conventional_ptychography --project-root .
```

Module form after editable install:

```bash
python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
```

Repo-local module form without installation:

```bash
PYTHONPATH=src python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
```

This interface is manifest-driven and now also gated by a confirmed contract
plus `judge_spec.ready=true` and `runtime_env.ready=true`. New tasks should not
require framework code changes in order to run live; they should enter through
registration and then reuse the same live runner.
