"""
demo.py
End-to-end demonstration of the from-scratch Spark RDD implementation.

Demonstrates:
  1. Narrow transformations (map, filter, flatMap) — pipelined in one stage
  2. Wide transformation (groupByKey, reduceByKey) — triggers shuffle + 2 stages
  3. Lazy evaluation — transformations build a DAG, actions trigger execution
  4. Lineage graph — printed to show the DAG structure
  5. Stage decomposition — scheduler splits at shuffle boundaries
  6. Fault tolerance via lineage — simulated partition failure and retry
"""
from context import SparkContext


def demo_narrow_transformations():
    print("=" * 70)
    print("DEMO 1: Narrow Transformations (map, filter, flatMap)")
    print("=" * 70)
    print("Narrow deps: each parent partition → at most one child partition.")
    print("The scheduler pipelines these into a SINGLE stage.\n")

    sc = SparkContext(num_workers=2)
    data = ["hello world", "spark rdd", "lineage graph", "lazy evaluation"]
    rdd = sc.parallelize(data, num_partitions=2)

    # Narrow pipeline: map → flatMap → filter
    words = (
        rdd
        .flatMap(lambda line: line.split(" "))
        .map(lambda word: word.lower())
        .filter(lambda word: len(word) > 3)
    )

    print("Lineage DAG:")
    words.print_dag()

    result = words.collect()
    print(f"\nResult: {result}")
    print(f"Count: {words.count()}\n")


def demo_wide_transformation():
    print("=" * 70)
    print("DEMO 2: Wide Transformation (groupByKey)")
    print("=" * 70)
    print("Wide deps: parent partition → many child partitions (shuffle).")
    print("The scheduler splits into TWO stages at the shuffle boundary.\n")

    sc = SparkContext(num_workers=2)
    data = [("a", 1), ("b", 2), ("a", 3), ("c", 4), ("b", 5), ("a", 6)]
    rdd = sc.parallelize(data, num_partitions=2)

    grouped = rdd.groupByKey(num_partitions=2)

    print("Lineage DAG:")
    grouped.print_dag()

    result = grouped.collect()
    print(f"\nResult (grouped by key):")
    for key, values in sorted(result):
        print(f"  {key}: {values}")
    print()


def demo_reduce_by_key():
    print("=" * 70)
    print("DEMO 3: reduceByKey")
    print("=" * 70)

    sc = SparkContext(num_workers=2)
    data = [("apple", 1), ("banana", 1), ("apple", 1), ("banana", 1), ("apple", 1)]
    rdd = sc.parallelize(data, num_partitions=2)

    counts = rdd.reduceByKey(lambda a, b: a + b)

    print("Lineage DAG:")
    counts.print_dag()

    result = counts.collect()
    print(f"\nResult (reduced by key):")
    for key, value in sorted(result):
        print(f"  {key}: {value}")
    print()


def demo_lazy_evaluation():
    print("=" * 70)
    print("DEMO 4: Lazy Evaluation")
    print("=" * 70)
    print("Transformations build a DAG but do NOT execute until an action.\n")

    sc = SparkContext(num_workers=2)
    rdd = sc.parallelize([1, 2, 3, 4, 5], num_partitions=2)

    # These do nothing except build the lineage graph
    mapped = rdd.map(lambda x: x * 2)
    filtered = mapped.filter(lambda x: x > 4)

    print("After transformations (no execution yet):")
    filtered.print_dag()

    # This triggers the DAG scheduler
    print("\nTriggering action: collect()")
    result = filtered.collect()
    print(f"Result: {result}\n")

    # Triggering another action recomputes from scratch (no caching)
    print("Triggering action again: count()")
    print("(Notice: the DAG is re-executed from scratch — no caching by default)")
    count = filtered.count()
    print(f"Count: {count}\n")


def demo_fault_tolerance():
    print("=" * 70)
    print("DEMO 5: Fault Tolerance via Lineage")
    print("=" * 70)
    print("If a partition is lost, we recompute it from the lineage graph.")
    print("No replication needed — the DAG itself is the recovery mechanism.\n")

    sc = SparkContext(num_workers=2)
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    rdd = sc.parallelize(data, num_partitions=2)

    # Build a simple pipeline
    doubled = rdd.map(lambda x: x * 2)

    print("Lineage DAG:")
    doubled.print_dag()

    print("\nSimulating a task failure on partition 0...")
    print("The executor will retry, recomputing the partition via lineage.\n")

    # We simulate failure by monkey-patching compute temporarily
    original_compute = rdd.compute
    call_count = [0]

    def failing_compute(partition_index):
        if partition_index == 0 and call_count[0] == 0:
            call_count[0] += 1
            raise RuntimeError("Simulated partition loss!")
        return original_compute(partition_index)

    rdd.compute = failing_compute

    try:
        result = doubled.collect()
        print(f"\nResult after recovery: {result}")
        print("The partition was recomputed from the lineage graph — no data was lost.\n")
    finally:
        rdd.compute = original_compute


def demo_word_count():
    print("=" * 70)
    print("DEMO 6: Word Count (map + flatMap + reduceByKey)")
    print("=" * 70)
    print("Classic MapReduce pattern using narrow + wide transformations.\n")

    sc = SparkContext(num_workers=2)
    lines = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "the fox is quick",
        "the dog is lazy"
    ]
    rdd = sc.parallelize(lines, num_partitions=2)

    word_counts = (
        rdd
        .flatMap(lambda line: line.split(" "))
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
    )

    print("Lineage DAG:")
    word_counts.print_dag()

    result = word_counts.collect()
    print(f"\nWord counts:")
    for word, count in sorted(result):
        print(f"  {word}: {count}")
    print()


if __name__ == "__main__":
    demo_narrow_transformations()
    demo_wide_transformation()
    demo_reduce_by_key()
    demo_lazy_evaluation()
    demo_fault_tolerance()
    demo_word_count()
