from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from run_video_smoke_test import (
    APP_DIR,
    SERVER_ALL_SETTINGS,
    SMOKE_OUTPUT_DIR,
    build_manifest_for_existing_videos,
    discover_repo_videos,
    ensure_dir,
    generate_smoke_videos,
    is_video_readable,
    open_video_frame,
    path_to_str,
    pick_warmup_frame,
    post_detect,
    short_windows_path,
    start_server,
    stop_process_tree,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay every smoke-video frame through the backend and compare legacy vs stable overlay behavior.")
    parser.add_argument("--run-name", default=f"overlay_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--manifest-path", type=Path, help="Optional manifest JSON to use instead of repo video discovery.")
    parser.add_argument("--compare-to", type=Path, help="Optional previous overlay_comparison.json to diff against.")
    return parser.parse_args()


def frame_annotation(sample_frames: list[dict[str, Any]], frame_index: int) -> dict[str, Any]:
    if not sample_frames:
        return {
            "expected_server_parts": [],
            "required_server_parts": [],
        }

    selected = sample_frames[0]
    for item in sample_frames:
        if int(item.get("frame_index", 0)) <= frame_index:
            selected = item
        else:
            break

    return {
        "expected_server_parts": sorted(set(selected.get("expected_server_parts") or [])),
        "required_server_parts": sorted(set(selected.get("required_server_parts") or [])),
    }


def collect_video_trace(video_spec: dict[str, Any]) -> dict[str, Any]:
    video_path = Path(video_spec["video_path"])
    cap = cv2.VideoCapture(short_windows_path(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlay trace: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    effective_fps = fps if fps > 0 else 6.0
    frames: list[dict[str, Any]] = []
    sample_frames = sorted(video_spec.get("sample_frames") or [], key=lambda item: int(item.get("frame_index", 0)))

    for frame_index in range(frame_count):
        frame = open_video_frame(video_path, frame_index)
        response = post_detect(frame, timeout=60.0)
        polygons = response.get("polygons", [])
        annotation = frame_annotation(sample_frames, frame_index)
        frames.append(
            {
                "frame_index": frame_index,
                "time_ms": round(frame_index * (1000.0 / effective_fps), 3),
                "polygons": polygons,
                "detected_parts": sorted({str(item.get("part", "")) for item in polygons if item.get("part")}),
                "expected_parts": annotation["expected_server_parts"],
                "required_parts": annotation["required_server_parts"],
            }
        )

    return {
        "video_name": video_spec["name"],
        "video_path": path_to_str(video_path),
        "fps": effective_fps,
        "frame_count": frame_count,
        "size": {"width": width, "height": height},
        "frames": frames,
    }


def build_summary_markdown(trace_path: Path, comparison: dict[str, Any]) -> str:
    lines = [
        "# Overlay Stability Smoke Report",
        "",
        f"- source_trace: {trace_path}",
        f"- legacy toggles: {comparison['overall']['toggles']['legacy']}",
        f"- stable toggles: {comparison['overall']['toggles']['stable']}",
        f"- legacy midstream toggles: {comparison['overall']['midstream_toggles']['legacy']}",
        f"- stable midstream toggles: {comparison['overall']['midstream_toggles']['stable']}",
        f"- legacy short gaps: {comparison['overall']['short_gaps']['legacy']}",
        f"- stable short gaps: {comparison['overall']['short_gaps']['stable']}",
        f"- legacy required miss frames: {comparison['overall']['required_miss_frames']['legacy']}",
        f"- stable required miss frames: {comparison['overall']['required_miss_frames']['stable']}",
        f"- legacy midstream required miss frames: {comparison['overall']['midstream_required_miss_frames']['legacy']}",
        f"- stable midstream required miss frames: {comparison['overall']['midstream_required_miss_frames']['stable']}",
        f"- legacy hold-after-loss frames: {comparison['overall']['hold_after_loss_frames']['legacy']}",
        f"- stable hold-after-loss frames: {comparison['overall']['hold_after_loss_frames']['stable']}",
        f"- legacy max hold-after-loss frames: {comparison['overall']['max_hold_after_loss_frames']['legacy']}",
        f"- stable max hold-after-loss frames: {comparison['overall']['max_hold_after_loss_frames']['stable']}",
        f"- legacy false reappearances: {comparison['overall']['false_reappearances']['legacy']}",
        f"- stable false reappearances: {comparison['overall']['false_reappearances']['stable']}",
        f"- legacy eye fragmented frames: {comparison['overall'].get('eye_fragmented_frames', {}).get('legacy', 0)}",
        f"- stable eye fragmented frames: {comparison['overall'].get('eye_fragmented_frames', {}).get('stable', 0)}",
        "",
    ]

    for video in comparison.get("videos", []):
        metrics = video["comparison"]
        lines.extend(
            [
                f"## {video['video_name']}",
                "",
                f"- toggles legacy -> stable: {metrics['total_toggles']['legacy']} -> {metrics['total_toggles']['stable']}",
                f"- midstream toggles legacy -> stable: {metrics['total_midstream_toggles']['legacy']} -> {metrics['total_midstream_toggles']['stable']}",
                f"- short gaps legacy -> stable: {metrics['total_short_gaps']['legacy']} -> {metrics['total_short_gaps']['stable']}",
                f"- required miss frames legacy -> stable: {metrics['required_miss_frames']['legacy']} -> {metrics['required_miss_frames']['stable']}",
                f"- midstream required miss frames legacy -> stable: {metrics['midstream_required_miss_frames']['legacy']} -> {metrics['midstream_required_miss_frames']['stable']}",
                f"- hold-after-loss frames legacy -> stable: {metrics['hold_after_loss_frames']['legacy']} -> {metrics['hold_after_loss_frames']['stable']}",
                f"- max hold-after-loss frames legacy -> stable: {metrics['max_hold_after_loss_frames']['legacy']} -> {metrics['max_hold_after_loss_frames']['stable']}",
                f"- false reappearances legacy -> stable: {metrics['false_reappearances']['legacy']} -> {metrics['false_reappearances']['stable']}",
                f"- eye fragmented frames legacy -> stable: {metrics.get('eye_fragmented_frames', {}).get('legacy', 0)} -> {metrics.get('eye_fragmented_frames', {}).get('stable', 0)}",
                "",
            ]
        )

    return "\n".join(lines)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_delta_report(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    metric_names = [
        "toggles",
        "midstream_toggles",
        "short_gaps",
        "required_miss_frames",
        "midstream_required_miss_frames",
        "hold_after_loss_frames",
        "max_hold_after_loss_frames",
        "false_reappearances",
        "eye_fragmented_frames",
    ]

    previous_videos = {item["video_name"]: item for item in previous.get("videos", [])}
    delta_videos: list[dict[str, Any]] = []

    for video in current.get("videos", []):
        previous_video = previous_videos.get(video["video_name"])
        if not previous_video:
            continue

        video_delta: dict[str, Any] = {"video_name": video["video_name"]}
        for metric_name in metric_names:
            previous_metric = previous_video["comparison"].get(metric_name)
            current_metric = video["comparison"].get(metric_name)
            if not previous_metric or not current_metric:
                continue

            video_delta[metric_name] = {
                "previous_stable": previous_metric["stable"],
                "current_stable": current_metric["stable"],
                "delta": current_metric["stable"] - previous_metric["stable"],
            }

        delta_videos.append(video_delta)

    overall_delta: dict[str, Any] = {}
    for metric_name in metric_names:
        previous_metric = previous.get("overall", {}).get(metric_name)
        current_metric = current.get("overall", {}).get(metric_name)
        if not previous_metric or not current_metric:
            continue
        overall_delta[metric_name] = {
            "previous_stable": previous_metric["stable"],
            "current_stable": current_metric["stable"],
            "delta": current_metric["stable"] - previous_metric["stable"],
        }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "previous_report": previous.get("source_trace") or previous.get("created_at"),
        "current_report": current.get("source_trace") or current.get("created_at"),
        "overall": overall_delta,
        "videos": delta_videos,
    }


def build_delta_markdown(delta: dict[str, Any]) -> str:
    lines = [
        "# Overlay Stability Delta",
        "",
        f"- previous_report: {delta['previous_report']}",
        f"- current_report: {delta['current_report']}",
        "",
    ]

    for metric_name, metric in delta.get("overall", {}).items():
        lines.append(f"- {metric_name}: {metric['previous_stable']} -> {metric['current_stable']} (delta {metric['delta']:+d})")

    lines.append("")

    for video in delta.get("videos", []):
        lines.append(f"## {video['video_name']}")
        lines.append("")
        for metric_name, metric in video.items():
            if metric_name == "video_name":
                continue
            lines.append(f"- {metric_name}: {metric['previous_stable']} -> {metric['current_stable']} (delta {metric['delta']:+d})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_dir = ensure_dir(SMOKE_OUTPUT_DIR / args.run_name)

    if args.manifest_path:
        manifest = load_manifest(args.manifest_path)
    else:
        repo_videos = [path for path in discover_repo_videos() if is_video_readable(path)]
        if repo_videos:
            manifest = build_manifest_for_existing_videos(repo_videos)
        else:
            manifest = generate_smoke_videos()

    manifest_path = run_dir / "manifest_used.json"
    write_json(manifest_path, manifest)
    warmup_frame = pick_warmup_frame(manifest)

    server_handle = None
    try:
        server_handle, backend_info = start_server(run_dir, warmup_frame)

        trace = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": path_to_str(run_dir),
            "backend": backend_info,
            "part_ids": sorted(SERVER_ALL_SETTINGS.keys()),
            "videos": [],
        }

        for video_spec in manifest["videos"]:
            trace["videos"].append(collect_video_trace(video_spec))

        trace_path = run_dir / "overlay_trace.json"
        write_json(trace_path, trace)

        node_bin = shutil.which("node.exe") or shutil.which("node")
        if not node_bin:
            raise RuntimeError("Node.js is not available on PATH for overlay replay analysis")

        comparison_path = run_dir / "overlay_comparison.json"
        result = subprocess.run(
            [node_bin, str(APP_DIR / "overlay_replay.js"), str(trace_path), str(comparison_path)],
            cwd=str(APP_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "overlay replay analysis failed")

        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        summary_path = run_dir / "overlay_comparison.md"
        summary_path.write_text(build_summary_markdown(trace_path, comparison), encoding="utf-8")

        delta_path = None
        delta_summary_path = None
        if args.compare_to:
            previous = json.loads(args.compare_to.read_text(encoding="utf-8"))
            delta = build_delta_report(previous, comparison)
            delta_path = run_dir / "overlay_delta.json"
            delta_summary_path = run_dir / "overlay_delta.md"
            write_json(delta_path, delta)
            delta_summary_path.write_text(build_delta_markdown(delta), encoding="utf-8")

        print(
            json.dumps(
                {
                    "run_dir": path_to_str(run_dir),
                    "trace_path": path_to_str(trace_path),
                    "comparison_path": path_to_str(comparison_path),
                    "summary_path": path_to_str(summary_path),
                    "delta_path": path_to_str(delta_path) if delta_path else None,
                    "delta_summary_path": path_to_str(delta_summary_path) if delta_summary_path else None,
                    "overall": comparison["overall"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if server_handle is not None:
            stop_process_tree(server_handle.process)


if __name__ == "__main__":
    main()
