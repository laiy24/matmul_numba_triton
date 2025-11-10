#!/usr/bin/env python3
"""Render a control-flow graph dump to an image."""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch


NodeId = int
Adjacency = Dict[NodeId, Iterable[NodeId]]
Position = Dict[NodeId, Tuple[float, float]]

LoopMarkers = Tuple[set[NodeId], set[NodeId], set[NodeId]]


def parse_sections(path: Path) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current_label: str | None = None
    buffer: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if line.endswith(":") and line.upper().startswith("CFG "):
            if current_label is not None:
                sections[current_label] = "\n".join(buffer).strip()
                buffer.clear()
            current_label = line[:-1]
        else:
            buffer.append(raw_line)
    if current_label is not None:
        sections[current_label] = "\n".join(buffer).strip()
    return sections


def parse_adjacency(text: str) -> Adjacency:
    raw = ast.literal_eval(text)
    adjacency: Dict[NodeId, Iterable[NodeId]] = {}
    for key, value in raw.items():
        adjacency[int(key)] = [int(v) for v in value]
    return adjacency


def parse_dominators(text: str) -> Dict[NodeId, set[NodeId]]:
    raw = ast.literal_eval(text)
    dominators: Dict[NodeId, set[NodeId]] = {}
    for key, value in raw.items():
        dominators[int(key)] = {int(v) for v in value}
    return dominators


def parse_int_set(text: str) -> set[NodeId]:
    text = text.strip()
    if not text:
        return set()
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def parse_loops(text: str) -> LoopMarkers:
    entries: set[NodeId] = set()
    exits: set[NodeId] = set()
    headers: set[NodeId] = set()
    pattern = re.compile(
        r"(\d+):\s*Loop\(entries=\{([^}]*)\},\s*exits=\{([^}]*)\},\s*header=(\d+)",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        entries.update(parse_int_set(match.group(2)))
        exits.update(parse_int_set(match.group(3)))
        headers.add(int(match.group(4)))
    return entries, exits, headers


def compute_positions(adjacency: Adjacency, dominators: Dict[NodeId, set[NodeId]] | None) -> Position:
    nodes: set[NodeId] = set(adjacency.keys())
    for successors in adjacency.values():
        nodes.update(int(s) for s in successors)

    level_map: Dict[int, list[NodeId]] = {}
    if dominators:
        for node in nodes:
            dom = dominators.get(node)
            depth = len(dom) - 1 if dom else 0
            level_map.setdefault(depth, []).append(node)
    else:
        # Fallback: assign by BFS depth from the entry node.
        if not nodes:
            return {}
        start = min(nodes)
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, depth = queue.pop(0)
            level_map.setdefault(depth, []).append(current)
            for nxt in adjacency.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
        # Any disconnected nodes get the deepest level.
        deepest = max(level_map)
        for node in nodes - visited:
            level_map.setdefault(deepest + 1, []).append(node)

    positions: Position = {}
    horizontal_gap = 2.0
    vertical_gap = 1.8
    for depth in sorted(level_map):
        layer = sorted(level_map[depth])
        count = len(layer)
        if count == 1:
            x_positions = [0.0]
        else:
            x_positions = [horizontal_gap * (idx - (count - 1) / 2) for idx in range(count)]
        for node, x in zip(layer, x_positions):
            positions[node] = (x, -depth * vertical_gap)
    return positions


def shrink_segment(src: Tuple[float, float], dst: Tuple[float, float], shrink: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return src, dst
    ratio = shrink / distance
    return (src[0] + dx * ratio, src[1] + dy * ratio), (dst[0] - dx * ratio, dst[1] - dy * ratio)


def draw_cfg(
    adjacency: Adjacency,
    positions: Position,
    output: Path,
    loop_entries: set[NodeId],
    loop_exits: set[NodeId],
    loop_headers: set[NodeId],
) -> None:
    if not positions:
        raise ValueError("No nodes to draw; the CFG appears to be empty.")

    node_radius = 0.6
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw edges first so that nodes stay on top.
    for src, targets in adjacency.items():
        for dst in targets:
            start = positions[src]
            end = positions[dst]
            if src == dst:
                loop_offset = node_radius * 2
                arrow = FancyArrowPatch(
                    (start[0], start[1] + loop_offset),
                    (start[0] + loop_offset, start[1]),
                    connectionstyle="arc3,rad=0.4",
                    arrowstyle="->",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="#555555",
                )
                ax.add_patch(arrow)
                continue
            trimmed_start, trimmed_end = shrink_segment(start, end, node_radius)
            arrow = FancyArrowPatch(
                trimmed_start,
                trimmed_end,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.2,
                color="#555555",
                connectionstyle="arc3,rad=0.1",
            )
            ax.add_patch(arrow)

    for node, (x, y) in positions.items():
        if node in loop_headers:
            fill_color = "#f57c00"
        elif node in loop_entries:
            fill_color = "#388e3c"
        elif node in loop_exits:
            fill_color = "#d32f2f"
        else:
            fill_color = "#1976d2"
        patch = Circle((x, y), node_radius, facecolor=fill_color, edgecolor="black", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            fontweight="semibold",
        )

    legend_handles: list[Patch] = []
    if loop_entries:
        legend_handles.append(Patch(facecolor="#388e3c", edgecolor="black", label="Loop entry"))
    if loop_headers:
        legend_handles.append(Patch(facecolor="#f57c00", edgecolor="black", label="Loop header"))
    if loop_exits:
        legend_handles.append(Patch(facecolor="#d32f2f", edgecolor="black", label="Loop exit"))
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(min(p[0] for p in positions.values()) - 1.5, max(p[0] for p in positions.values()) + 1.5)
    ax.set_ylim(min(p[1] for p in positions.values()) - 1.5, max(p[1] for p in positions.values()) + 1.5)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a CFG image from a NUMBA_DUMP_CFG text file.")
    parser.add_argument("cfg_dump", type=Path, help="Path to the CFG dump file (e.g. tile_demo_print_cfg.txt)")
    parser.add_argument("--output", type=Path, default=Path("cfg.png"), help="Image file to create (default: cfg.png)")
    args = parser.parse_args()

    sections = parse_sections(args.cfg_dump)
    adjacency_text = sections.get("CFG adjacency lists")
    if not adjacency_text:
        raise ValueError("The dump file does not contain 'CFG adjacency lists'.")

    adjacency = parse_adjacency(adjacency_text)
    dominators = None
    dom_text = sections.get("CFG dominators")
    if dom_text:
        dominators = parse_dominators(dom_text)

    loop_entries: set[NodeId] = set()
    loop_exits: set[NodeId] = set()
    loop_headers: set[NodeId] = set()
    loops_text = sections.get("CFG loops")
    if loops_text:
        loop_entries, loop_exits, loop_headers = parse_loops(loops_text)

    positions = compute_positions(adjacency, dominators)
    draw_cfg(adjacency, positions, args.output, loop_entries, loop_exits, loop_headers)


if __name__ == "__main__":
    main()
