"""H5: compare logged reward-term curves of v18range vs v18base (tensorboard).

Usage:
  cd luwu_mjlab && .venv/bin/python scripts/debug_compare_tb.py
"""

import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
  "v18range": "logs/rsl_rl/xgolite_v18range/2026-07-12_01-48-38_v18range_v1",
  "v18base": "logs/rsl_rl/xgolite_v18draft/2026-07-12_01-21-55_v18base_v1",
}

accs = {}
for label, d in RUNS.items():
  f = glob.glob(d + "/events.out.tfevents.*")[0]
  acc = EventAccumulator(f, size_guidance={"scalars": 0})
  acc.Reload()
  accs[label] = acc

tags = {label: set(acc.Tags()["scalars"]) for label, acc in accs.items()}
common = sorted(tags["v18range"] & tags["v18base"])
only_range = sorted(tags["v18range"] - tags["v18base"])
print("tags only in v18range:", only_range)
print()


def at_iters(acc, tag, iters):
  evs = acc.Scalars(tag)
  out = {}
  for it in iters:
    best = min(evs, key=lambda e: abs(e.step - it))
    out[it] = best.value
  return out


CHECK_ITERS = [100, 300, 600, 1000, 1499]
print(f"{'tag':52s}" + "".join(f"{it:>10d}" for it in CHECK_ITERS))
for tag in common:
  if not (
    tag.startswith("Episode_Reward/")
    or tag.startswith("Curriculum/")
    or tag in ("Train/mean_reward", "Train/mean_episode_length")
    or tag.startswith("Metrics/twist/")
  ):
    continue
  r = at_iters(accs["v18range"], tag, CHECK_ITERS)
  b = at_iters(accs["v18base"], tag, CHECK_ITERS)
  print(f"{tag:52s}" + "".join(f"{r[it]:>10.3f}" for it in CHECK_ITERS) + "  | v18range")
  print(f"{'':52s}" + "".join(f"{b[it]:>10.3f}" for it in CHECK_ITERS) + "  | v18base")

print()
for tag in only_range:
  if "urriculum" in tag or "grid" in tag or "seed" in tag:
    evs = accs["v18range"].Scalars(tag)
    pts = [0, len(evs) // 8, len(evs) // 4, len(evs) // 2, 3 * len(evs) // 4, -1]
    s = ", ".join(f"it{evs[p].step}={evs[p].value:.4f}" for p in pts)
    print(f"{tag}: {s}")
# v18range trained 2500 iters: show its late-stage tracking too.
print()
for tag in (
  "Episode_Reward/track_linear_velocity",
  "Episode_Reward/track_angular_velocity",
  "Train/mean_reward",
):
  evs = accs["v18range"].Scalars(tag)
  late = [e for e in evs if e.step in (1800, 2100, 2400, 2499)]
  print(f"v18range late {tag}: " + ", ".join(f"it{e.step}={e.value:.3f}" for e in late))
