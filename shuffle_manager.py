"""
shuffle_manager.py
In-memory shuffle data storage.

During a wide dependency (ShuffleDependency), the parent stage
(Map Stage) writes shuffle data, and the child stage (Reduce Stage)
reads it.

In production Spark, shuffle data is spilled to disk for fault
tolerance. Here we keep it in memory for simplicity.
"""
import threading


class ShuffleManager:
    """
    Stores shuffle output keyed by (shuffle_id, reduce_partition).

    Map tasks write:  shuffle_manager.write(shuffle_id, reduce_partition, record)
    Reduce tasks read: shuffle_manager.read(shuffle_id, reduce_partition)
    """
    def __init__(self):
        self._data = {}
        self._lock = threading.Lock()

    def write(self, shuffle_id, partition, record):
        key = (shuffle_id, partition)
        with self._lock:
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(record)

    def read(self, shuffle_id, partition):
        key = (shuffle_id, partition)
        with self._lock:
            return list(self._data.get(key, []))

    def clear(self):
        with self._lock:
            self._data.clear()
