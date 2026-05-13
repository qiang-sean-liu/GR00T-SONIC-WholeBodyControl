"""Generate a Markdown data-quality report for a LeRobot dataset.

Quality/status standard used by this script:

- PASS:
  - Required parquet file exists.
  - Required columns exist.
  - Required numeric columns have no NaN/Inf.
  - Row count matches `meta/episodes.jsonl`.
  - Video files are readable and frame count matches parquet row count.
  - Timestamps are increasing and close to `1 / fps`.
  - No warnings are triggered.

- WARN:
  - Episode is listed in `meta/info.json` `discarded_episode_indices`.
  - Episode is shorter than `--min-frames` (default: 100).
  - Scene XML is missing or unreadable.
  - Timestamp jitter exceeds `--timestamp-tolerance` (default: 0.005s).
  - Any exact repeated transfer-payload frame is found.
    A repeated transfer-payload frame means frame N has the same payload as
    frame N-1 after excluding metadata columns such as `timestamp`,
    `frame_index`, `index`, and `task_index`. This is suspicious for stale
    ZMQ/DDS transfer. The current standard is conservative: even one exact
    repeated payload frame triggers WARN.

- FAIL:
  - Required parquet file is missing.
  - Required columns are missing.
  - Required numeric columns contain NaN/Inf.
  - Parquet row count differs from `meta/episodes.jsonl` length.
  - Required video file is missing or unreadable.

Example:

    python gear_sonic/scripts/validate_dataset_quality.py \
        --dataset-dir outputs/2026-05-11-00-31-20 \
        --episodes all

    python gear_sonic/scripts/validate_dataset_quality.py \
        --dataset-dir outputs/2026-05-11-00-31-20 \
        --episodes 9,17,22,24 \
        --output outputs/2026-05-11-00-31-20/data_quality_report_selected.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pyarrow.parquet as pq


KEY_ARRAY_COLUMNS = (
    "observation.state",
    "observation.eef_state",
    "action.wbc",
    "observation.root_orientation",
    "observation.projected_gravity",
    "teleop.smpl_pose",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
    "teleop.vr_3pt_position",
    "teleop.vr_3pt_orientation",
)

TRANSFER_PAYLOAD_COLUMNS = (
    "observation.state",
    "observation.eef_state",
    "action.wbc",
    "observation.root_orientation",
    "observation.projected_gravity",
    "action.motion_token",
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.body_quat_w",
    "teleop.target_body_orientation",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
    "teleop.left_wrist_joints",
    "teleop.right_wrist_joints",
    "teleop.vr_3pt_position",
    "teleop.vr_3pt_orientation",
)

REQUIRED_COLUMNS = (
    "observation.state",
    "action.wbc",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


@dataclass
class ColumnStats:
    name: str
    present: bool
    shape: str = "-"
    nan_count: int = 0
    inf_count: int = 0
    zero_rows: int | None = None
    repeated_rows: int | None = None
    repeated_frame_ranges: list[tuple[int, int]] = field(default_factory=list)
    max_step_l2: float | None = None


@dataclass
class EpisodeReport:
    episode_index: int
    expected_length: int | None = None
    parquet_path: Path | None = None
    parquet_exists: bool = False
    parquet_rows: int | None = None
    video_paths: dict[str, Path] = field(default_factory=dict)
    video_status: dict[str, str] = field(default_factory=dict)
    scene_xml_path: Path | None = None
    scene_xml_status: str = "missing"
    scene_objects: dict[str, list[float]] = field(default_factory=dict)
    discarded: bool = False
    missing_columns: list[str] = field(default_factory=list)
    column_stats: list[ColumnStats] = field(default_factory=list)
    repeated_payload_rows: int = 0
    repeated_payload_ranges: list[tuple[int, int]] = field(default_factory=list)
    repeated_payload_columns: list[str] = field(default_factory=list)
    timestamp_summary: str = "-"
    frame_index_summary: str = "-"
    episode_index_summary: str = "-"
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failures:
            return "FAIL"
        if self.warnings or self.discarded:
            return "WARN"
        return "PASS"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_parquet_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    return dataset_dir / pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def get_video_keys(info: dict) -> list[str]:
    keys = info.get("video_keys", [])
    if keys:
        return list(keys)
    return [
        key
        for key, spec in info.get("features", {}).items()
        if spec.get("dtype") in {"video", "image"}
    ]


def get_video_paths(dataset_dir: Path, info: dict, episode_index: int) -> dict[str, Path]:
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return {
        key: dataset_dir
        / pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=key,
        )
        for key in get_video_keys(info)
    }


def parse_episode_selection(selection: str, available: list[int]) -> list[int]:
    selection = selection.strip().lower()
    if selection in {"all", "*"}:
        return available

    selected: set[int] = set()
    for token in selection.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            selected.update(range(min(start, end), max(start, end) + 1))
        else:
            selected.add(int(token))
    return sorted(selected)


def table_column_to_numpy(table: pq.Table, column_name: str) -> np.ndarray:
    values = table[column_name].combine_chunks().to_pylist()
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(-1)
    return arr


def count_nan_inf(arr: np.ndarray) -> tuple[int, int]:
    if not np.issubdtype(arr.dtype, np.number):
        return 0, 0
    arr64 = arr.astype(np.float64, copy=False)
    return int(np.isnan(arr64).sum()), int(np.isinf(arr64).sum())


def build_ranges(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
            continue
        ranges.append((start, prev))
        start = current
        prev = current
    ranges.append((start, prev))
    return ranges


def format_ranges(ranges: list[tuple[int, int]], limit: int = 12) -> str:
    if not ranges:
        return "-"
    shown = []
    for start, end in ranges[:limit]:
        shown.append(str(start) if start == end else f"{start}-{end}")
    if len(ranges) > limit:
        shown.append(f"... +{len(ranges) - limit} ranges")
    return ", ".join(shown)


def format_repeated_pairs(ranges: list[tuple[int, int]], limit: int = 12) -> str:
    """Format repeated current-frame indices as previous->current pairs.

    A repeated index N means frame N equals frame N-1.  For a range A-B, every
    frame from A through B equals its previous frame, so the frozen span starts
    at A-1 and ends at B.
    """
    if not ranges:
        return "-"
    shown = []
    for start, end in ranges[:limit]:
        if start == end:
            shown.append(f"{start - 1}->{start}")
        else:
            shown.append(f"{start - 1}->{end}")
    if len(ranges) > limit:
        shown.append(f"... +{len(ranges) - limit} ranges")
    return ", ".join(shown)


def summarize_array_column(name: str, table: pq.Table) -> ColumnStats:
    if name not in table.column_names:
        return ColumnStats(name=name, present=False)

    arr = table_column_to_numpy(table, name)
    nan_count, inf_count = count_nan_inf(arr)
    stats = ColumnStats(
        name=name,
        present=True,
        shape="x".join(str(dim) for dim in arr.shape),
        nan_count=nan_count,
        inf_count=inf_count,
    )
    if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(arr.shape[0], -1).astype(np.float64, copy=False)
        stats.zero_rows = int(np.all(np.isclose(flat, 0.0), axis=1).sum())
        if len(flat) > 1:
            diffs = flat[1:] - flat[:-1]
            step_l2 = np.linalg.norm(diffs, axis=1)
            repeated_indices = np.where(np.all(flat[1:] == flat[:-1], axis=1))[0] + 1
            stats.repeated_rows = int(len(repeated_indices))
            stats.repeated_frame_ranges = build_ranges(repeated_indices)
            stats.max_step_l2 = float(np.nanmax(step_l2))
        else:
            stats.repeated_rows = 0
            stats.max_step_l2 = 0.0
    return stats


def detect_repeated_transfer_payload(table: pq.Table) -> tuple[int, list[tuple[int, int]], list[str]]:
    """Find frames whose transfer payload exactly matches the previous frame.

    Metadata columns such as timestamp, frame_index, index, and task_index are
    intentionally excluded because they can advance even when a ZMQ/DDS payload
    is stale.
    """
    if table.num_rows < 2:
        return 0, [], []

    repeated_mask = np.ones(table.num_rows - 1, dtype=bool)
    used_columns: list[str] = []
    for name in TRANSFER_PAYLOAD_COLUMNS:
        if name not in table.column_names:
            continue
        arr = table_column_to_numpy(table, name)
        if len(arr) != table.num_rows:
            continue
        if arr.ndim == 1:
            flat = arr.reshape(-1, 1)
        else:
            flat = arr.reshape(arr.shape[0], -1)
        repeated_mask &= np.all(flat[1:] == flat[:-1], axis=1)
        used_columns.append(name)

    if not used_columns:
        return 0, [], []

    repeated_indices = np.where(repeated_mask)[0] + 1
    return int(len(repeated_indices)), build_ranges(repeated_indices), used_columns


def summarize_timestamp(table: pq.Table, expected_fps: float, tolerance: float) -> tuple[str, list[str]]:
    if "timestamp" not in table.column_names:
        return "missing", ["missing timestamp column"]

    ts = table_column_to_numpy(table, "timestamp").astype(np.float64).reshape(-1)
    nan_count, inf_count = count_nan_inf(ts)
    warnings: list[str] = []
    if nan_count or inf_count:
        warnings.append(f"timestamp has {nan_count} NaN and {inf_count} Inf values")
    if len(ts) < 2:
        return f"frames={len(ts)}, not enough timestamps for dt check", warnings

    dt = np.diff(ts)
    expected_dt = 1.0 / expected_fps if expected_fps > 0 else float("nan")
    backwards = int((dt <= 0).sum())
    bad_dt = int((np.abs(dt - expected_dt) > tolerance).sum())
    if backwards:
        warnings.append(f"timestamp is non-increasing in {backwards} places")
    if bad_dt:
        warnings.append(f"{bad_dt} timestamp intervals differ from {expected_dt:.6f}s by > {tolerance:.6f}s")
    return (
        f"dt median={np.nanmedian(dt):.6f}s, min={np.nanmin(dt):.6f}s, "
        f"max={np.nanmax(dt):.6f}s, bad_dt={bad_dt}, non_increasing={backwards}",
        warnings,
    )


def summarize_index_columns(table: pq.Table, episode_index: int) -> tuple[str, str, list[str]]:
    warnings: list[str] = []

    if "frame_index" in table.column_names:
        frame_index = table_column_to_numpy(table, "frame_index").astype(np.int64).reshape(-1)
        expected = np.arange(len(frame_index), dtype=np.int64)
        bad = int((frame_index != expected).sum())
        frame_summary = f"start={frame_index[0] if len(frame_index) else '-'}, end={frame_index[-1] if len(frame_index) else '-'}, non_sequential={bad}"
        if bad:
            warnings.append(f"frame_index is not 0..N-1 in {bad} rows")
    else:
        frame_summary = "missing"
        warnings.append("missing frame_index column")

    if "episode_index" in table.column_names:
        episode_arr = table_column_to_numpy(table, "episode_index").astype(np.int64).reshape(-1)
        wrong = int((episode_arr != episode_index).sum())
        episode_summary = f"expected={episode_index}, wrong_rows={wrong}"
        if wrong:
            warnings.append(f"episode_index column has {wrong} rows not equal to {episode_index}")
    else:
        episode_summary = "missing"
        warnings.append("missing episode_index column")

    return frame_summary, episode_summary, warnings


def inspect_video(video_path: Path, expected_rows: int | None, expected_fps: float) -> str:
    if not video_path.exists():
        return "missing"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "unreadable"
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    row_msg = "unknown"
    if expected_rows is not None:
        diff = frames - expected_rows
        row_msg = f"diff_vs_rows={diff:+d}"
    fps_msg = "unknown"
    if math.isfinite(fps) and fps > 0:
        fps_msg = f"{fps:.2f}"
    return f"frames={frames}, fps={fps_msg}, {row_msg}"


def inspect_scene_xml(scene_xml_path: Path) -> tuple[str, dict[str, list[float]]]:
    if not scene_xml_path.exists():
        return "missing", {}
    try:
        tree = ET.parse(scene_xml_path)
    except ET.ParseError as exc:
        return f"parse_error: {exc}", {}

    objects: dict[str, list[float]] = {}
    for body_name in ("plate_body", "cube_body"):
        body = tree.find(f".//body[@name='{body_name}']")
        if body is None:
            objects[body_name] = []
            continue
        pos_s = body.attrib.get("pos")
        if not pos_s:
            objects[body_name] = []
            continue
        try:
            objects[body_name] = [float(x) for x in pos_s.split()]
        except ValueError:
            objects[body_name] = []
    return "present", objects


def inspect_episode(
    dataset_dir: Path,
    info: dict,
    meta_by_episode: dict[int, dict],
    discarded: set[int],
    episode_index: int,
    fps: float,
    timestamp_tolerance: float,
    min_frames: int,
) -> EpisodeReport:
    report = EpisodeReport(
        episode_index=episode_index,
        expected_length=meta_by_episode.get(episode_index, {}).get("length"),
        discarded=episode_index in discarded,
    )
    if report.discarded:
        report.warnings.append("episode is listed in info.json discarded_episode_indices")

    report.parquet_path = get_parquet_path(dataset_dir, info, episode_index)
    report.parquet_exists = report.parquet_path.exists()
    if not report.parquet_exists:
        report.failures.append(f"missing parquet: {report.parquet_path}")
        return report

    table = pq.read_table(report.parquet_path)
    report.parquet_rows = table.num_rows
    if report.expected_length is not None and report.parquet_rows != report.expected_length:
        report.failures.append(
            f"parquet rows ({report.parquet_rows}) != episodes.jsonl length ({report.expected_length})"
        )
    if report.parquet_rows < min_frames:
        report.warnings.append(f"short episode: {report.parquet_rows} frames < min_frames {min_frames}")

    report.missing_columns = [col for col in REQUIRED_COLUMNS if col not in table.column_names]
    if report.missing_columns:
        report.failures.append(f"missing required columns: {', '.join(report.missing_columns)}")

    (
        report.repeated_payload_rows,
        report.repeated_payload_ranges,
        report.repeated_payload_columns,
    ) = detect_repeated_transfer_payload(table)
    if report.repeated_payload_rows:
        report.warnings.append(
            f"{report.repeated_payload_rows} exact repeated transfer-payload frames"
        )

    for name in KEY_ARRAY_COLUMNS:
        stats = summarize_array_column(name, table)
        report.column_stats.append(stats)
        if stats.present and (stats.nan_count or stats.inf_count):
            report.failures.append(f"{name} has {stats.nan_count} NaN and {stats.inf_count} Inf values")

    timestamp_summary, timestamp_warnings = summarize_timestamp(table, fps, timestamp_tolerance)
    report.timestamp_summary = timestamp_summary
    report.warnings.extend(timestamp_warnings)

    frame_summary, episode_summary, index_warnings = summarize_index_columns(table, episode_index)
    report.frame_index_summary = frame_summary
    report.episode_index_summary = episode_summary
    report.warnings.extend(index_warnings)

    report.video_paths = get_video_paths(dataset_dir, info, episode_index)
    for key, path in report.video_paths.items():
        status = inspect_video(path, report.parquet_rows, fps)
        report.video_status[key] = status
        if status in {"missing", "unreadable"}:
            report.failures.append(f"video {key} is {status}: {path}")
        elif "diff_vs_rows=+0" not in status and "diff_vs_rows=-0" not in status:
            report.warnings.append(f"video {key} frame count differs from parquet rows ({status})")

    report.scene_xml_path = dataset_dir / "scene_xml" / f"episode_{episode_index:06d}.xml"
    report.scene_xml_status, report.scene_objects = inspect_scene_xml(report.scene_xml_path)
    if report.scene_xml_status != "present":
        report.warnings.append(f"scene XML is {report.scene_xml_status}: {report.scene_xml_path}")

    return report


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


def write_report(
    output_path: Path,
    dataset_dir: Path,
    info: dict,
    selected: list[int],
    reports: list[EpisodeReport],
) -> None:
    passed = sum(r.status == "PASS" for r in reports)
    warned = sum(r.status == "WARN" for r in reports)
    failed = sum(r.status == "FAIL" for r in reports)
    discarded = sorted(info.get("discarded_episode_indices", []))

    lines: list[str] = []
    lines.append("# Dataset Quality Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_dir}`")
    lines.append(f"- Selected episodes: `{','.join(str(i) for i in selected)}`")
    lines.append(f"- FPS: `{info.get('fps', '-')}`")
    lines.append(f"- Total episodes in metadata: `{info.get('total_episodes', '-')}`")
    lines.append(f"- Total frames in metadata: `{info.get('total_frames', '-')}`")
    lines.append(f"- Discarded episodes in metadata: `{discarded}`")
    lines.append(f"- Result: `{passed}` pass, `{warned}` warn, `{failed}` fail")
    lines.append("")

    lines.append("## Repeated Frames Summary")
    lines.append("")
    repeated_reports = [report for report in reports if report.repeated_payload_rows > 0]
    repeated_columns = sorted(
        {
            column
            for report in reports
            for column in report.repeated_payload_columns
        }
    )
    if repeated_reports:
        lines.append(
            "This section detects exact full-payload repeats: frame `N` has the same recorded "
            "transfer payload as frame `N-1`, excluding metadata columns like `timestamp`, "
            "`frame_index`, `index`, and `task_index`. These are the frames most suspicious for "
            "failed or stale ZMQ/DDS transfer."
        )
        lines.append("")
        lines.append(f"- Payload columns checked: `{', '.join(repeated_columns)}`")
        lines.append("")
        lines.append("| Episode | Exact Repeated Payload Frames | Previous->Current Frame Pairs |")
        lines.append("|---:|---:|---|")
        for report in repeated_reports:
            lines.append(
                f"| {report.episode_index} | {report.repeated_payload_rows} "
                f"| `{format_repeated_pairs(report.repeated_payload_ranges)}` |"
            )
    else:
        lines.append(
            "No exact repeated transfer-payload frames were found. Per-column repeated ranges "
            "are still listed in each episode's detailed column table."
        )
    lines.append("")

    lines.append("## Episode Summary")
    lines.append("")
    lines.append("| Episode | Status | Rows | Metadata Length | Discarded | Main Issues |")
    lines.append("|---:|---|---:|---:|---|---|")
    for report in reports:
        issues = report.failures or report.warnings
        issue_text = "<br>".join(issues[:4]) if issues else "-"
        if len(issues) > 4:
            issue_text += f"<br>... {len(issues) - 4} more"
        lines.append(
            f"| {report.episode_index} | {report.status} | {report.parquet_rows if report.parquet_rows is not None else '-'} "
            f"| {report.expected_length if report.expected_length is not None else '-'} | {report.discarded} | {issue_text} |"
        )
    lines.append("")

    for report in reports:
        lines.append(f"## Episode {report.episode_index:06d}")
        lines.append("")
        lines.append(f"- Status: `{report.status}`")
        lines.append(f"- Parquet: `{report.parquet_path}`")
        lines.append(f"- Rows: `{report.parquet_rows}`")
        lines.append(f"- Expected length: `{report.expected_length}`")
        lines.append(f"- Timestamp: {report.timestamp_summary}")
        lines.append(f"- Frame index: {report.frame_index_summary}")
        lines.append(f"- Episode index: {report.episode_index_summary}")
        lines.append(
            f"- Exact repeated transfer-payload frames: `{report.repeated_payload_rows}` "
            f"({format_repeated_pairs(report.repeated_payload_ranges)})"
        )
        if report.failures:
            lines.append(f"- Failures: {'; '.join(report.failures)}")
        if report.warnings:
            lines.append(f"- Warnings: {'; '.join(report.warnings)}")
        lines.append("")

        lines.append("### Files")
        lines.append("")
        lines.append("| Artifact | Status | Path |")
        lines.append("|---|---|---|")
        lines.append(
            f"| parquet | {'present' if report.parquet_exists else 'missing'} | `{report.parquet_path}` |"
        )
        for key, status in report.video_status.items():
            lines.append(f"| video `{key}` | {status} | `{report.video_paths[key]}` |")
        lines.append(f"| scene_xml | {report.scene_xml_status} | `{report.scene_xml_path}` |")
        lines.append("")

        if report.scene_objects:
            lines.append("### Scene Objects")
            lines.append("")
            lines.append("| Body | Position |")
            lines.append("|---|---|")
            for name, pos in report.scene_objects.items():
                pos_s = ", ".join(f"{v:.4f}" for v in pos) if pos else "-"
                lines.append(f"| `{name}` | `{pos_s}` |")
            lines.append("")

        lines.append("### Column Checks")
        lines.append("")
        lines.append(
            "| Column | Present | Shape | NaN | Inf | Zero Rows | Repeated Rows | Repeated Frame Ranges | Max Step L2 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|")
        for stats in report.column_stats:
            lines.append(
                f"| `{stats.name}` | {stats.present} | {stats.shape} | {stats.nan_count} | {stats.inf_count} "
                f"| {stats.zero_rows if stats.zero_rows is not None else '-'} "
                f"| {stats.repeated_rows if stats.repeated_rows is not None else '-'} "
                f"| `{format_ranges(stats.repeated_frame_ranges)}` "
                f"| {format_float(stats.max_step_l2)} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="LeRobot dataset root.")
    parser.add_argument(
        "--episodes",
        default="all",
        help="Episode selection: all, comma list, or ranges, e.g. 9,17,22-24.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path. Default: <dataset-dir>/data_quality_report.md.",
    )
    parser.add_argument("--min-frames", type=int, default=100, help="Warn if an episode has fewer frames.")
    parser.add_argument(
        "--timestamp-tolerance",
        type=float,
        default=0.005,
        help="Warn when timestamp dt differs from 1/fps by more than this many seconds.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes_meta = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    meta_by_episode = {int(row["episode_index"]): row for row in episodes_meta}
    available = sorted(meta_by_episode)
    selected = parse_episode_selection(args.episodes, available)
    discarded = {int(x) for x in info.get("discarded_episode_indices", [])}
    fps = float(info.get("fps", 50))

    reports = [
        inspect_episode(
            dataset_dir=dataset_dir,
            info=info,
            meta_by_episode=meta_by_episode,
            discarded=discarded,
            episode_index=episode_index,
            fps=fps,
            timestamp_tolerance=args.timestamp_tolerance,
            min_frames=args.min_frames,
        )
        for episode_index in selected
    ]

    output_path = args.output or (dataset_dir / "data_quality_report.md")
    write_report(output_path, dataset_dir, info, selected, reports)

    passed = sum(r.status == "PASS" for r in reports)
    warned = sum(r.status == "WARN" for r in reports)
    failed = sum(r.status == "FAIL" for r in reports)
    print(f"Wrote {output_path}")
    print(f"Result: {passed} pass, {warned} warn, {failed} fail")


if __name__ == "__main__":
    main()
