from dependency import ShuffleDependency, OneToOneDependency

"""
scheduler.py
DAG Scheduler — decomposes an action into Stages.

From the paper:
  "The scheduler examines the RDD's lineage graph to build a DAG of stages.
   Each stage contains pipelined transformations with narrow dependencies.
   Stage boundaries are the shuffle operations required for wide dependencies."

Execution flow:
  1. Action triggers run_job(final_rdd, func).
  2. Scheduler walks backward to build stages.
  3. Stages are submitted in topological order (parents before children).
  4. ShuffleMapStages write shuffle data.
  5. ResultStage computes the final action result.
"""


class Stage:
    """Base class for a stage in the DAG."""
    def __init__(self, stage_id, rdd, num_partitions):
        self.id = stage_id
        self.rdd = rdd
        self.num_partitions = num_partitions
        self.parents = []  # parent stages that must complete first

    def __repr__(self):
        return f"Stage({self.id}, {self.rdd.__class__.__name__}, partitions={self.num_partitions})"


class ShuffleMapStage(Stage):
    """
    A stage that computes partitions of an RDD and writes shuffle data.
    This stage is the parent of a wide dependency (ShuffleDependency).
    """
    def __init__(self, stage_id, rdd, shuffle_dep, num_partitions):
        super().__init__(stage_id, rdd, num_partitions)
        self.shuffle_dep = shuffle_dep

    def __repr__(self):
        return f"ShuffleMapStage({self.id}, shuffle_id={self.shuffle_dep.shuffle_id})"


class ResultStage(Stage):
    """
    The final stage that computes the action result.
    Each task applies `func` to a partition of the stage's RDD.
    """
    def __init__(self, stage_id, rdd, func, num_partitions):
        super().__init__(stage_id, rdd, num_partitions)
        self.func = func

    def __repr__(self):
        return f"ResultStage({self.id}, action_func)"


class DAGScheduler:
    def __init__(self, executor):
        self.executor = executor
        self._next_stage_id = 0

    def new_stage_id(self):
        sid = self._next_stage_id
        self._next_stage_id += 1
        return sid

    def run_job(self, final_rdd, func):
        """
        Main entry point. Build the stage DAG and execute it.
        Returns a list of results, one per partition of the final RDD.
        """
        result_stage = self._create_result_stage(final_rdd, func)
        all_stages = self._get_all_stages(result_stage)

        print("\n[Scheduler] Stage plan:")
        for i, stage in enumerate(all_stages):
            print(f"  Stage {i}: {stage}")

        # Execute stages in dependency order
        for stage in all_stages:
            if isinstance(stage, ShuffleMapStage):
                print(f"\n[Scheduler] Running ShuffleMapStage {stage.id} (shuffle_id={stage.shuffle_dep.shuffle_id})")
                self.executor.run_shuffle_map_stage(stage)
            elif isinstance(stage, ResultStage):
                print(f"\n[Scheduler] Running ResultStage {stage.id}")
                return self.executor.run_result_stage(stage)

    def _create_result_stage(self, rdd, func):
        """Build a ResultStage and recursively discover parent ShuffleMapStages."""
        parent_stages = []
        visited_shuffles = set()

        def visit(current_rdd):
            for dep in current_rdd.get_dependencies():
                if isinstance(dep, ShuffleDependency):
                    if dep.shuffle_id not in visited_shuffles:
                        visited_shuffles.add(dep.shuffle_id)
                        parent_stage = self._create_shuffle_map_stage(dep.rdd, dep)
                        parent_stages.append(parent_stage)
                else:
                    # Narrow dependency — keep walking backward
                    visit(dep.rdd)

        visit(rdd)
        stage = ResultStage(self.new_stage_id(), rdd, func, rdd.num_partitions)
        stage.parents = parent_stages
        return stage

    def _create_shuffle_map_stage(self, rdd, shuffle_dep):
        """Build a ShuffleMapStage and recursively discover its parents."""
        parent_stages = []
        visited_shuffles = set()

        def visit(current_rdd):
            for dep in current_rdd.get_dependencies():
                if isinstance(dep, ShuffleDependency):
                    if dep.shuffle_id not in visited_shuffles:
                        visited_shuffles.add(dep.shuffle_id)
                        parent_stage = self._create_shuffle_map_stage(dep.rdd, dep)
                        parent_stages.append(parent_stage)
                else:
                    visit(dep.rdd)

        visit(rdd)
        stage = ShuffleMapStage(self.new_stage_id(), rdd, shuffle_dep, rdd.num_partitions)
        stage.parents = parent_stages
        return stage

    def _get_all_stages(self, stage):
        """Return stages in topological order (parents first)."""
        result = []
        visited = set()

        def add_stages(s):
            if s.id in visited:
                return
            visited.add(s.id)
            for p in s.parents:
                add_stages(p)
            result.append(s)

        add_stages(stage)
        return result
