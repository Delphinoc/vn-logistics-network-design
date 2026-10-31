"""Stage 1: build demand nodes and candidate hubs for mainland Vietnam."""
import _setup  # noqa: F401
from vnnet.data import build_nodes
from vnnet.paths import PROCESSED

nodes, dropped = build_nodes()
nodes.to_csv(PROCESSED / "nodes.csv", index=False)
dropped.to_csv(PROCESSED / "merged_districts.csv", index=False)
print(f"{len(nodes)} demand nodes ({nodes.is_candidate_hub.sum()} candidate hubs), "
      f"population {nodes.population.sum():,}; {len(dropped)} districts merged into parent cities")
