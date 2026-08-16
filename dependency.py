"""
dependency.py
Narrow vs Wide Dependencies — the core distinction from the 2012 paper.

Narrow:  parent partition → at most one child partition (pipelined)
Wide:   parent partition → many child partitions (requires shuffle)
"""


class Dependency:
    """Base class for all dependencies."""
    pass


class NarrowDependency(Dependency):
    """
    Each parent partition contributes to at most one child partition.
    Examples: map, filter, flatMap.
    These can be pipelined in a single thread without network I/O.
    """
    def __init__(self, rdd):
        self.rdd = rdd

    def get_parents(self, partition_index):
        """Return the parent partition index(es) for a given child partition."""
        raise NotImplementedError


class OneToOneDependency(NarrowDependency):
    """
    Partition i of parent → partition i of child.
    Used by map, filter, flatMap, etc.
    """
    def get_parents(self, partition_index):
        return [partition_index]


class ShuffleDependency(Dependency):
    """
    Wide dependency — requires a shuffle phase across the cluster.
    Multiple child partitions may depend on data from a single parent partition.
    Examples: groupByKey, reduceByKey, sortByKey.

    The scheduler inserts a Stage boundary at every ShuffleDependency.
    """
    def __init__(self, rdd, num_partitions, shuffle_id, partitioner):
        self.rdd = rdd
        self.num_partitions = num_partitions
        self.shuffle_id = shuffle_id
        self.partitioner = partitioner
