"""
executor.py
Task executor with a thread pool.

From the paper:
  "The scheduler sends tasks to workers, which compute partitions
   and either store them in memory or write shuffle output."

Tasks:
  - ShuffleMapTask: compute a partition and write shuffle data.
  - ResultTask: compute a partition and apply the action function.

Fault tolerance via lineage is demonstrated by retrying failed tasks.
Because each task recomputes its partition from the lineage graph,
a lost partition can always be reconstructed.
"""
import concurrent.futures
from dependency import ShuffleDependency, OneToOneDependency


class Task:
    """Base class for a unit of work sent to a worker."""
    def __init__(self, stage, partition_index):
        self.stage = stage
        self.partition_index = partition_index
        self.task_id = f"{stage.id}_{partition_index}"

    def run(self, shuffle_manager):
        raise NotImplementedError


class ShuffleMapTask(Task):
    """
    Computes one partition of a ShuffleMapStage's RDD,
    partitions records by key, and writes to the shuffle manager.
    """
    def run(self, shuffle_manager):
        iterator = self.stage.rdd.compute(self.partition_index)
        shuffle_dep = self.stage.shuffle_dep
        count = 0
        for record in iterator:
            # record must be (key, value) for key-based shuffle
            key = record[0]
            reduce_partition = shuffle_dep.partitioner.get_partition(
                key, shuffle_dep.num_partitions
            )
            shuffle_manager.write(shuffle_dep.shuffle_id, reduce_partition, record)
            count += 1
        return count


class ResultTask(Task):
    """
    Computes one partition of a ResultStage's RDD
    and applies the action function.
    """
    def run(self, shuffle_manager):
        iterator = self.stage.rdd.compute(self.partition_index)
        return self.stage.func(iterator)


class Executor:
    def __init__(self, num_workers, shuffle_manager):
        self.num_workers = num_workers
        self.shuffle_manager = shuffle_manager

    def run_shuffle_map_stage(self, stage):
        """Run all tasks in a ShuffleMapStage."""
        tasks = [ShuffleMapTask(stage, i) for i in range(stage.num_partitions)]
        self._run_tasks(tasks)
        print(f"  [Executor] ShuffleMapStage {stage.id} complete. Wrote shuffle_id={stage.shuffle_dep.shuffle_id}")

    def run_result_stage(self, stage):
        """Run all tasks in a ResultStage and return results in partition order."""
        tasks = [ResultTask(stage, i) for i in range(stage.num_partitions)]
        results = self._run_tasks(tasks)
        print(f"  [Executor] ResultStage {stage.id} complete.")
        return results

    def _run_tasks(self, tasks):
        """Execute tasks in parallel with retry logic."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {pool.submit(self._run_task_with_retry, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                result = future.result()
                results[task.partition_index] = result
        # Return in partition order
        return [results[i] for i in range(len(tasks))]

    def _run_task_with_retry(self, task, max_retries=2):
        """Run a task, retrying on failure to demonstrate lineage-based recovery."""
        for attempt in range(max_retries):
            try:
                return task.run(self.shuffle_manager)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Task {task.task_id} failed after {max_retries} attempts: {e}"
                    )
                print(f"  [Retry] Task {task.task_id} failed (attempt {attempt + 1}), "
                      f"recomputing via lineage...")
