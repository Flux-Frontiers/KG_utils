# Release Notes — v0.20.0

> Released: 2026-09-08

### Added

Five extension points on `SnapshotManager`, each replacing an override that
two or more KG modules were carrying. Nothing here changes existing behaviour:
every default reproduces what the base did in 0.19.1.

- **`package_name` as a class attribute.** Seven of the eight KG modules carry
  an `__init__` whose entire body is `super().__init__(snapshots_dir,
  package_name="...", db_path=db_path)`. A subclass now sets the attribute and
  deletes the method. An explicit `package_name=` keyword still wins for a
  single instance.

- **`_domain_metrics(stats)`, a capture hook.** Modules that collect their own
  metrics -- per-module node counts, per-directory counts, topic counts --
  override `capture()` to do it, which means restating the base signature.
  Restating it is how an unnamed `key=` fell into `**extra_metrics` and shipped
  four packages keyed on a tree hash. Collecting metrics in `_domain_metrics()`
  leaves the signature alone, so that failure cannot recur. It receives the
  graph stats, so a module can also derive a metric from them, and the values
  it returns yield to a same-named `extra_metrics` keyword, so it can declare
  a domain default that an explicit caller overrides.

- **`timestamp` in the `diff_snapshots` result.** `doc_kg` and `memory_kg`
  carried byte-identical `diff_snapshots` overrides that existed for nothing
  but this, at the cost of two extra `load_snapshot` calls each.

- **`issues_delta` in the `diff_snapshots` result.** `issues` is a base field
  and "introduced / resolved" needs no domain knowledge. Ordered by appearance
  in the source list rather than by set iteration, so the result is stable
  across runs.

- **`dict_metric_deltas`, a class attribute.** `pycode_kg`
  (`module_node_counts`), `ftree_kg` (`dir_node_counts`) and `diary_kg`
  (`topic_counts`) each hand-rolled the same loop: diff a dict-valued metric,
  keep only the changed keys. Naming the metric keys on the subclass emits
  `"<key>_delta"` for each.

- **`metrics_ignore`, a class attribute.** `doc_kg`'s `_metrics_changed`
  override exists only to drop `db_path` before comparing. Empty by default,
  so `_metrics_changed` is unchanged for every other module.

- **`capture_aliases`, a class attribute.** `capture()` ends in
  `**extra_metrics`, so it accepts any keyword. A module that renames one of
  its own capture keywords therefore gets no error from the old name: the value
  lands in the metrics dict under the dead name and the new one is simply
  absent. That is the same silence that shipped four packages keyed on a tree
  hash. Declaring `{old: new}` keeps the old keyword working and raises a
  `DeprecationWarning` naming the replacement. The target may be a metric name
  or `graph_stats_dict`; an explicitly passed current keyword always wins.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
