"""Check deliverables and recorded evidence. Run from any working directory.

Requires Python 3, ffprobe, pdfinfo; add --decode-videos for full ffmpeg decode.
"""
import argparse
import ast
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-videos", action="store_true")
    args = ap.parse_args()
    required = ["README.md", "SUBMISSION_CHECKLIST.md", "task1_rosorin/README.md",
                "task1_rosorin/src", "task1_rosorin/configs", "task1_rosorin/maps",
                "task2_mobile_manipulation/README.md", "task2_mobile_manipulation/hommi_repro",
                "task2_mobile_manipulation/momanipvla_repro", "task2_mobile_manipulation/scripts",
                "task2_mobile_manipulation/results", "task2_mobile_manipulation/checkpoints/hommi_latest.ckpt",
                "report/technical_report.pdf", "videos/part1_demo.mp4", "videos/part2_demo.mp4",
                "videos/part2_commands.mp4", "task2_mobile_manipulation/scripts/run_command_demo.sh"]
    for name in required:
        p = ROOT / name
        assert p.exists() and (p.is_dir() or p.stat().st_size > 0), f"Missing/empty: {name}"
    links = 0
    for name in ["README.md", "SUBMISSION_CHECKLIST.md", "task1_rosorin/README.md",
                 "task2_mobile_manipulation/README.md", "videos/README.md"]:
        p = ROOT / name
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if re.match(r"[a-zA-Z]+:", target) or target.startswith("#"):
                continue
            dest = unquote(target.split("#")[0].strip("<>"))
            assert (p.parent / dest).exists(), f"Broken link in {name}: {target}"
            links += 1
    for p in ROOT.rglob("*"):
        assert not p.is_symlink(), f"Symlink is not standalone: {p.relative_to(ROOT)}"
    t1 = ROOT / "task1_rosorin"
    for exposed, runtime in [("configs", "config"), ("maps", "maps")]:
        for p in (t1 / exposed).rglob("*"):
            if p.is_file():
                q = t1 / "src/robot_gazebo" / runtime / p.relative_to(t1 / exposed)
                assert q.exists() and p.read_bytes() == q.read_bytes(), f"Different runtime copy: {p}"
    routes = {}
    for folder, patterns in [("full_slam", ["single_goal.json", "waypoints.json", "explore_*.json"]),
                             ("controller_check", ["single_goal.json", "waypoints.json", "uturn_waypoints.json"])]:
        for pattern in patterns:
            paths = sorted((t1 / "evidence" / folder).glob(pattern))
            assert paths, (folder, pattern)
            for p in paths:
                d = json.loads(p.read_text())
                assert d["passed"] and d["action_status"] == 4 and not d["missed_waypoints"], p.name
                assert d["settled_after_result"] and len(d["arrivals"]) == len(d["poses"])
                tol = d["acceptance_tolerances"]
                for i, pose in enumerate(d["poses"]):
                    measured = d["arrivals"][str(i)]
                    # Route files store poses as [x,y,yaw].
                    x, y, yaw = pose if isinstance(pose, list) else (pose["x"], pose["y"], pose["yaw"])
                    xy = math.hypot(measured["x"]-x, measured["y"]-y)
                    dyaw = abs(math.atan2(math.sin(measured["yaw"]-yaw), math.cos(measured["yaw"]-yaw)))
                    assert xy <= tol["xy_m"] and dyaw <= tol["yaw_rad"]
                    assert math.isclose(xy, measured["xy_error_m"], abs_tol=1e-7)
                assert d["final_error"]["xy_m"] <= tol["xy_m"] and d["final_error"]["yaw_rad"] <= tol["yaw_rad"]
                routes[f"{folder}/{p.name}"] = dict(waypoints=len(d["poses"]), final_error=d["final_error"])
    report = ROOT / "report/technical_report.pdf"
    info = subprocess.check_output(["pdfinfo", str(report)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1])
    assert 5 <= pages <= 7, f"Report length: {pages}"
    body = subprocess.check_output(["pdftotext", str(report), "-"], text=True)
    assert not re.search(r"\b(TODO|TBD|placeholder)\b", body, re.I), "Unfinished report text"
    media = {}
    for name in ["part1_demo.mp4", "part2_demo.mp4", "part2_commands.mp4"]:
        p = ROOT / "videos" / name
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format",
                           "-show_streams", "-of", "json", str(p)], text=True))
        videos = [s for s in probe["streams"] if s["codec_type"] == "video"]
        assert videos and videos[0]["width"] > 0
        duration = float(probe["format"]["duration"])
        assert duration > 0
        if name == "part1_demo.mp4":
            assert 180 <= duration <= 300
        if args.decode_videos:
            subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(p), "-f", "null", "-"], check=True)
        media[p.name] = dict(duration_s=duration, codec=videos[0]["codec_name"],
                              width=videos[0]["width"], height=videos[0]["height"],
                              full_decode=args.decode_videos)
    compiled = 0
    for folder in ["task1_rosorin/scripts", "task1_rosorin/src", "task2_mobile_manipulation/hommi_repro",
                   "task2_mobile_manipulation/momanipvla_repro", "task2_mobile_manipulation/scripts",
                   "task2_mobile_manipulation/tests", "report"]:
        for p in (ROOT / folder).rglob("*.py"):
            ast.parse(p.read_text(), filename=str(p))
            compiled += 1
    recording = json.loads((ROOT / "task2_mobile_manipulation/results/command_demo/recording.json").read_text())
    assert recording["command_exit_status"] == 0
    output = dict(status="PASS", report_pages=pages, videos=media, navigation=routes,
                  command_recording=recording,
                  local_links_checked=links, python_files_parsed=compiled,
                  scope="Package and recorded-evidence checks; no new full ROS run or full retraining")
    (ROOT / "report/validation.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
