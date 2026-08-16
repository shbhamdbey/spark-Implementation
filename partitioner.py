"""
partitioner.py
Determines how records are routed across partitions during a shuffle.

From the paper: "We use a hash-based partitioner by default, 
but users can supply custom partitioners."
"""


class Partitioner:
    """Base class for partitioning strategies."""
    def get_partition(self, key, num_partitions):
        raise NotImplementedError


class HashPartitioner(Partitioner):
    """
    Default partitioner used by groupByKey / reduceByKey.
    Routes records to partitions based on hash(key) % num_partitions.
    """
    def get_partition(self, key, num_partitions):
        return hash(key) % num_partitions
