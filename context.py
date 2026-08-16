"""
context.py
SparkContext — the entry point for RDD operations.

Manages:
  - RDD and shuffle ID counters
  - The DAG scheduler
  - The task executor
  - The shuffle manager
"""
from shuffle_manager import ShuffleManager
from scheduler import DAGScheduler
from executor import Executor
from rdd import ParallelCollectionRDD


class SparkContext:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.shuffle_manager = ShuffleManager()
        self.executor = Executor(num_workers, self.shuffle_manager)
        self.scheduler = DAGScheduler(self.executor)
        self._rdd_id = 0
        self._shuffle_id = 0

    def new_rdd_id(self):
        rid = self._rdd_id
        self._rdd_id += 1
        return rid

    def new_shuffle_id(self):
        sid = self._shuffle_id
        self._shuffle_id += 1
        return sid

    def parallelize(self, data, num_partitions=None):
        """Create an RDD from a local Python collection."""
        if num_partitions is None:
            num_partitions = self.num_workers
        return ParallelCollectionRDD(self, data, num_partitions)

    def run_action(self, rdd, func):
        """Trigger execution of an action on the given RDD."""
        return self.scheduler.run_job(rdd, func)
