"""
rdd.py
Resilient Distributed Dataset base class and transformations.

Core idea from the 2012 paper:
- An RDD is a read-only, partitioned collection of records.
- Fault tolerance is achieved through lineage (a graph of operations),
  not replication.
- Transformations are LAZY — they build a DAG but do not execute.
- Actions trigger the DAG scheduler to run the job.
"""
from functools import reduce
from dependency import OneToOneDependency, ShuffleDependency
from partitioner import HashPartitioner


class RDD:
    """
    Resilient Distributed Dataset.

    Each RDD knows:
      - its partitions
      - its parent RDDs (dependencies)
      - how to compute a partition from its parents
    """
    def __init__(self, context, num_partitions, dependencies=None):
        self.context = context
        self.num_partitions = num_partitions
        self.dependencies = dependencies or []
        self.id = context.new_rdd_id()

    def get_partitions(self):
        return list(range(self.num_partitions))

    def compute(self, partition_index):
        """
        Compute a specific partition.
        Subclasses override this to define the transformation logic.
        """
        raise NotImplementedError

    def get_dependencies(self):
        return self.dependencies

    # ------------------------------------------------------------------
    # Narrow Transformations (pipelined, no shuffle)
    # ------------------------------------------------------------------
    def map(self, f):
        """Return a new RDD by applying f to each element."""
        return MapRDD(self.context, self, f)

    def filter(self, f):
        """Return a new RDD containing only elements satisfying f."""
        return FilterRDD(self.context, self, f)

    def flatMap(self, f):
        """Return a new RDD by flattening the results of f."""
        return FlatMapRDD(self.context, self, f)

    def glom(self):
        """Return an RDD where each element is a list of its partition."""
        return GlomRDD(self.context, self)

    # ------------------------------------------------------------------
    # Wide Transformations (shuffle boundary)
    # ------------------------------------------------------------------
    def groupByKey(self, num_partitions=None):
        """
        Group values by key. Requires a shuffle.
        Input must be an RDD of (key, value) pairs.
        """
        return GroupByKeyRDD(self.context, self, num_partitions)

    def reduceByKey(self, f, num_partitions=None):
        """
        Merge values for each key using f. Requires a shuffle.
        Input must be an RDD of (key, value) pairs.
        """
        return ReduceByKeyRDD(self.context, self, f, num_partitions)

    # ------------------------------------------------------------------
    # Actions (trigger execution)
    # ------------------------------------------------------------------
    def collect(self):
        """Return all elements as a list at the driver."""
        results = self.context.run_action(self, lambda iterator: list(iterator))
        return [item for partition in results for item in partition]

    def count(self):
        """Return the total number of elements."""
        results = self.context.run_action(self, lambda iterator: sum(1 for _ in iterator))
        return sum(results)

    def reduce(self, f):
        """Aggregate elements using f."""
        results = self.context.run_action(
            self, lambda iterator: self._reduce_partition(f, iterator)
        )
        # Filter out None (empty partitions)
        valid = [r for r in results if r is not None]
        if not valid:
            raise ValueError("Cannot reduce empty RDD")
        return reduce(f, valid)

    def take(self, n):
        """Return the first n elements."""
        results = self.context.run_action(
            self, lambda iterator: self._take_partition(iterator, n)
        )
        taken = []
        for partition_result in results:
            for item in partition_result:
                taken.append(item)
                if len(taken) >= n:
                    return taken[:n]
        return taken

    def _reduce_partition(self, f, iterator):
        it = iter(iterator)
        try:
            result = next(it)
        except StopIteration:
            return None
        for item in it:
            result = f(result, item)
        return result

    def _take_partition(self, iterator, n):
        result = []
        for item in iterator:
            result.append(item)
            if len(result) >= n:
                break
        return result

    # ------------------------------------------------------------------
    # Debugging / Lineage
    # ------------------------------------------------------------------
    def print_dag(self, indent=0):
        """Print the lineage graph (DAG) rooted at this RDD."""
        prefix = "  " * indent
        print(f"{prefix}└─ RDD {self.id} ({self.__class__.__name__}), partitions={self.num_partitions}")
        for dep in self.dependencies:
            if isinstance(dep, OneToOneDependency):
                dep.rdd.print_dag(indent + 1)
            elif isinstance(dep, ShuffleDependency):
                print(f"{prefix}   └─ [SHUFFLE shuffle_id={dep.shuffle_id}] →")
                dep.rdd.print_dag(indent + 2)


# ==================================================================
# Concrete RDD Implementations
# ==================================================================

class ParallelCollectionRDD(RDD):
    """
    Starting RDD created from a Python list.
    Splits the data into num_partitions slices.
    """
    def __init__(self, context, data, num_partitions):
        super().__init__(context, num_partitions)
        self.data = data
        self.slice_size = max(1, len(data) // num_partitions) if data else 1

    def compute(self, partition_index):
        start = partition_index * self.slice_size
        if partition_index == self.num_partitions - 1:
            end = len(self.data)
        else:
            end = start + self.slice_size
        return iter(self.data[start:end])


class MapRDD(RDD):
    """Narrow: apply a function to each element."""
    def __init__(self, context, parent, f):
        super().__init__(context, parent.num_partitions, [OneToOneDependency(parent)])
        self.parent = parent
        self.f = f

    def compute(self, partition_index):
        return (self.f(x) for x in self.parent.compute(partition_index))


class FilterRDD(RDD):
    """Narrow: keep only elements satisfying a predicate."""
    def __init__(self, context, parent, f):
        super().__init__(context, parent.num_partitions, [OneToOneDependency(parent)])
        self.parent = parent
        self.f = f

    def compute(self, partition_index):
        return (x for x in self.parent.compute(partition_index) if self.f(x))


class FlatMapRDD(RDD):
    """Narrow: map then flatten."""
    def __init__(self, context, parent, f):
        super().__init__(context, parent.num_partitions, [OneToOneDependency(parent)])
        self.parent = parent
        self.f = f

    def compute(self, partition_index):
        for x in self.parent.compute(partition_index):
            for y in self.f(x):
                yield y


class GlomRDD(RDD):
    """Narrow: coalesce each partition into a single list element."""
    def __init__(self, context, parent):
        super().__init__(context, parent.num_partitions, [OneToOneDependency(parent)])
        self.parent = parent

    def compute(self, partition_index):
        return iter([list(self.parent.compute(partition_index))])


class GroupByKeyRDD(RDD):
    """
    Wide: group values by key. Creates a ShuffleDependency.

    Stage decomposition:
      Stage 1 (ShuffleMap): parent RDD → partition by key → write shuffle data
      Stage 2 (Result): read shuffle data → group by key
    """
    def __init__(self, context, parent, num_partitions=None):
        if num_partitions is None:
            num_partitions = parent.num_partitions
        partitioner = HashPartitioner()
        shuffle_id = context.new_shuffle_id()
        super().__init__(
            context, num_partitions,
            [ShuffleDependency(parent, num_partitions, shuffle_id, partitioner)]
        )
        self.shuffle_id = shuffle_id

    def compute(self, partition_index):
        shuffle_data = self.context.shuffle_manager.read(self.shuffle_id, partition_index)
        groups = {}
        for key, value in shuffle_data:
            groups.setdefault(key, []).append(value)
        return iter(groups.items())


class ReduceByKeyRDD(RDD):
    """
    Wide: reduce values by key. Creates a ShuffleDependency.

    This is a simplified version without map-side combine.
    Stage 1 shuffles (key, value) pairs.
    Stage 2 groups and reduces.
    """
    def __init__(self, context, parent, f, num_partitions=None):
        if num_partitions is None:
            num_partitions = parent.num_partitions
        partitioner = HashPartitioner()
        shuffle_id = context.new_shuffle_id()
        super().__init__(
            context, num_partitions,
            [ShuffleDependency(parent, num_partitions, shuffle_id, partitioner)]
        )
        self.shuffle_id = shuffle_id
        self.f = f

    def compute(self, partition_index):
        shuffle_data = self.context.shuffle_manager.read(self.shuffle_id, partition_index)
        groups = {}
        for key, value in shuffle_data:
            if key in groups:
                groups[key] = self.f(groups[key], value)
            else:
                groups[key] = value
        return iter(groups.items())
