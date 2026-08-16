# Spark from Scratch — Resilient Distributed Datasets (RDDs)

> **Paper:** *"Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing"*  
> **Authors:** Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Justin Ma, Murphy McCauley, Michael Franklin, Scott Shenker, Ion Stoica (UC Berkeley, 2012)  
> **Links:** [ACM DL](https://dl.acm.org/doi/10.5555/2228298.2228301) | [USENIX ATC](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)

---

## What I Built

I implemented a **minimal but fully working** Spark-like distributed computing engine in pure Python, built ground-up from the core design principles described in the 2012 RDD paper. The system consists of:

- An **RDD base class** (`rdd.py`) with lineage tracking and lazy evaluation
- **Narrow transformations** (`map`, `filter`, `flatMap`) that are pipelined
- **Wide transformations** (`groupByKey`, `reduceByKey`) that trigger shuffle stages
- A **DAG Scheduler** (`scheduler.py`) that decomposes jobs into stages at shuffle boundaries
- A **Task Executor** (`executor.py`) that runs tasks in a thread pool with retry logic
- A **Shuffle Manager** (`shuffle_manager.py`) for wide-dependency data exchange
- An **end-to-end demo** (`demo.py`) showing narrow deps, wide deps, lazy evaluation, and fault tolerance

Everything runs with **zero external dependencies** — only the Python standard library.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DRIVER PROCESS                               │
│  1. Parses DAG    2. Splits into Stages    3. Schedules Tasks       │
└─────────────────────────────────────────────────────────────────────┘
                             │                   │
            ┌────────────────┘                   └────────────────┐
            │ Task (ShuffleMap)                                │ Task (Result)
            v                                                    v
┌───────────────────────┐                        ┌───────────────────────┐
│    WORKER 1           │                        │    WORKER 2           │
│  ┌─────────────────┐  │                        │  ┌─────────────────┐  │
│  │ Partition 0     │  │      Shuffle Write     │  │ Partition 1     │  │
│  │  ┌───────────┐  │  │◄──────────────────────►│  │  ┌───────────┐  │  │
│  │  │ map(...)  │  │  │      (wide dep)        │  │  │ map(...)  │  │  │
│  │  │ filter(..)│  │  │                        │  │  │ filter(..)│  │  │
│  │  └───────────┘  │  │                        │  │  └───────────┘  │  │
│  └─────────────────┘  │                        │  └─────────────────┘  │
└───────────────────────┘                        └───────────────────────┘
```

---

## Core Design Principles from the Paper — What I Implemented

### 1. Lineage Graph (DAG)
> *"Instead of checkpointing intermediate data to disk, store the sequence of operations (transformations) used to build a dataset. If a partition is lost, recompute only that partition."*

**My implementation:** `rdd.py` — `RDD` base class

- Every RDD stores its **parent RDDs** in `self.dependencies`.
- Every RDD knows how to **compute a partition** via `compute(partition_index)`.
- The lineage is a directed acyclic graph (DAG) of operations, not a copy of the data.
- If a task fails, the executor **retries** it — the task recomputes the partition from scratch by walking the lineage graph.

```python
# rdd.py
class RDD:
    def __init__(self, context, num_partitions, dependencies=None):
        self.dependencies = dependencies or []  # lineage graph edges

    def compute(self, partition_index):
        raise NotImplementedError  # each RDD knows how to rebuild itself
```

This is the paper's central insight: **fault tolerance through lineage, not replication**.

---

### 2. Lazy Evaluation
> *"Transformations construct an Execution Graph (DAG). Computation is deferred until an action is triggered."*

**My implementation:** `rdd.py` — transformation methods

- `map()`, `filter()`, `flatMap()`, `groupByKey()` do **nothing** except create a new RDD node pointing to the parent.
- No data is processed until an **action** (`collect()`, `count()`, `reduce()`) is called.
- The action calls `SparkContext.run_action()`, which invokes the DAGScheduler.

```python
# rdd.py — transformations are lazy
def map(self, f):
    return MapRDD(self.context, self, f)  # just creates a node, no compute

def collect(self):
    # This triggers the scheduler
    results = self.context.run_action(self, lambda iterator: list(iterator))
    return [item for partition in results for item in partition]
```

---

### 3. Narrow vs Wide Dependencies
> *"Narrow: each parent partition is used by at most one child partition. Wide: multiple child partitions depend on data from a single parent partition."*

**My implementation:** `dependency.py`

| Type | Class | Example | Scheduler Behavior |
|------|-------|---------|-------------------|
| **Narrow** | `OneToOneDependency` | `map`, `filter`, `flatMap` | Pipelined into a **single stage** |
| **Wide** | `ShuffleDependency` | `groupByKey`, `reduceByKey` | Creates a **stage boundary** + shuffle |

```python
# dependency.py
class OneToOneDependency(NarrowDependency):
    def get_parents(self, partition_index):
        return [partition_index]  # partition i → partition i

class ShuffleDependency(Dependency):
    def __init__(self, rdd, num_partitions, shuffle_id, partitioner):
        self.rdd = rdd
        self.num_partitions = num_partitions
        self.shuffle_id = shuffle_id
        self.partitioner = partitioner  # routes records across partitions
```

---

### 4. DAG Scheduler — Stage Decomposition
> *"The scheduler walks backward from an Action to decompose the job into Stages. Stage boundaries are at shuffle operations."*

**My implementation:** `scheduler.py` — `DAGScheduler`

The scheduler builds stages by walking backward from the action RDD:

1. **ResultStage**: the final stage that applies the action function (e.g., `list(iterator)`).
2. **ShuffleMapStage**: intermediate stages that compute parent partitions and write shuffle data.

```python
# scheduler.py
class DAGScheduler:
    def run_job(self, final_rdd, func):
        result_stage = self._create_result_stage(final_rdd, func)
        all_stages = self._get_all_stages(result_stage)

        for stage in all_stages:
            if isinstance(stage, ShuffleMapStage):
                self.executor.run_shuffle_map_stage(stage)
            elif isinstance(stage, ResultStage):
                return self.executor.run_result_stage(stage)
```

For a pipeline like `parallelize → map → filter → groupByKey → map → collect`, the scheduler produces:

```
Stage 0: ShuffleMapStage
  └─ parallelize → map → filter → (write shuffle data)

Stage 1: ResultStage
  └─ (read shuffle data) → map → collect
```

---

### 5. Pipelined Execution of Narrow Transformations
> *"Narrow dependencies can be executed in a single pipelined thread without materializing intermediate records."*

**My implementation:** `rdd.py` — `MapRDD.compute()`, `FilterRDD.compute()`

- `MapRDD.compute()` returns a **generator** that lazily pulls from the parent.
- `FilterRDD.compute()` returns a **generator** that lazily filters from the parent.
- Multiple narrow transformations are **fused** into a single task — no intermediate arrays are allocated.

```python
# rdd.py
class MapRDD(RDD):
    def compute(self, partition_index):
        return (self.f(x) for x in self.parent.compute(partition_index))

class FilterRDD(RDD):
    def compute(self, partition_index):
        return (x for x in self.parent.compute(partition_index) if self.f(x))
```

---

### 6. Shuffle for Wide Dependencies
> *"Wide dependencies require a shuffle phase across network nodes."*

**My implementation:** `shuffle_manager.py` + `executor.py`

- **ShuffleMapTask**: computes a parent partition, hashes each record by key, and writes to `ShuffleManager`.
- **ResultTask**: reads the pre-shuffled data for its partition from `ShuffleManager`.
- The `HashPartitioner` routes `(key, value)` pairs to reduce partitions using `hash(key) % num_partitions`.

```python
# executor.py — ShuffleMapTask
class ShuffleMapTask(Task):
    def run(self, shuffle_manager):
        for record in self.stage.rdd.compute(self.partition_index):
            key = record[0]
            reduce_partition = shuffle_dep.partitioner.get_partition(key, ...)
            shuffle_manager.write(shuffle_dep.shuffle_id, reduce_partition, record)
```

---

### 7. Fault Tolerance via Lineage
> *"If a partition is lost, the scheduler recomputes it by re-executing the tasks that built it."*

**My implementation:** `executor.py` — `_run_task_with_retry()`

- Each task is executed with **retry logic** (max 2 attempts).
- On failure, the task is re-submitted. Because the task recomputes the partition from the lineage graph, no data is lost.
- This demonstrates the paper's claim: **lineage eliminates the need for replication**.

```python
# executor.py
def _run_task_with_retry(self, task, max_retries=2):
    for attempt in range(max_retries):
        try:
            return task.run(self.shuffle_manager)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [Retry] Task {task.task_id} failed, recomputing via lineage...")
```

---

## Project Structure

```
spark-from-scratch/
├── dependency.py        # NarrowDependency, OneToOneDependency, ShuffleDependency
├── partitioner.py       # HashPartitioner
├── shuffle_manager.py   # In-memory shuffle data storage
├── rdd.py               # RDD base class + MapRDD, FilterRDD, GroupByKeyRDD, etc.
├── scheduler.py         # DAGScheduler (stage decomposition)
├── executor.py          # Executor with thread pool + task retry
├── context.py           # SparkContext (entry point)
├── demo.py              # End-to-end demonstrations
└── README.md            # This file
```

---

## Quick Start

### Run the full demo
```bash
python demo.py
```

Output includes:
- Narrow transformation pipeline (single stage)
- Wide transformation with shuffle (two stages)
- Lazy evaluation proof (DAG built but not executed)
- Fault tolerance demo (simulated failure + lineage recovery)
- Word count using map + flatMap + reduceByKey

### Interactive usage
```python
from context import SparkContext

sc = SparkContext(num_workers=4)

# Narrow pipeline (single stage)
rdd = sc.parallelize([1, 2, 3, 4, 5])
doubled = rdd.map(lambda x: x * 2)
filtered = doubled.filter(lambda x: x > 4)
print(filtered.collect())  # [6, 8, 10]

# Wide transformation (shuffle, two stages)
pairs = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])
grouped = pairs.groupByKey()
print(grouped.collect())   # [("a", [1, 3]), ("b", [2])]

# View the lineage graph
filtered.print_dag()
```

---

## What I Did NOT Implement (Honest Scope)

| Feature | Paper Status | Why I Skipped It |
|---------|-------------|------------------|
| **Multiple physical nodes** | Core concept | Single-process with threads; network is simulated in-memory |
| **Memory caching / persistence** | Supported (`cache()`, `persist()`) | Omitted to focus on lineage; trivial to add |
| **Custom partitioners** | Supported | Only `HashPartitioner` implemented |
| **Map-side combine** | Optimization for `reduceByKey` | Simplified to shuffle-then-reduce |
| **Broadcast variables / accumulators** | Additional APIs | Out of scope for core RDD demo |
| **Spark SQL / DataFrames** | Later additions to Spark | RDD-only, as per the 2012 paper |

---

## Why This Matters

The 2012 RDD paper introduced a fundamental shift in distributed computing:

1. **Don't replicate data for fault tolerance** — track lineage instead.
2. **Don't execute transformations eagerly** — build a DAG and optimize the whole plan.
3. **Don't materialize intermediate results** — pipeline narrow transformations with generators.
4. **Don't treat shuffle as an afterthought** — make it a first-class stage boundary.

My implementation proves these principles can be expressed in ~600 lines of Python and actually work. The DAG scheduler correctly splits at shuffle boundaries, narrow transformations are pipelined without intermediate storage, and lineage-based retry recovers from failures without any data replication.

---

## License

MIT — Educational use only.
