"""Build the portable curriculum and physically silent demonstration videos.

Run once when lesson content changes; the learning app never needs the source
dataset or FFmpeg. Original clips are read without modification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]

# Display/answer translations do not change the original model class IDs.
TRANSLATIONS = [
    "magandang umaga", "magandang hapon", "magandang gabi", "kumusta|hi|kamusta",
    "kumusta ka|kamusta ka", "mabuti ako|i am fine|i'm fine|ayos lang ako",
    "ikinagagalak kitang makilala|masaya akong makilala ka", "salamat|thanks",
    "walang anuman|you're welcome|you are welcome", "kita tayo bukas|magkita tayo bukas",
    "naiintindihan|naintindihan", "hindi naiintindihan|dont understand|do not understand",
    "alam", "hindi alam|dont know|do not know", "hindi", "oo|opo", "mali|incorrect", "tama|right",
    "mabagal|dahan dahan", "mabilis", "isa|1", "dalawa|2", "tatlo|3", "apat|4", "lima|5",
    "anim|6", "pito|7", "walo|8", "siyam|9", "sampu|10",
    "enero", "pebrero", "marso", "abril", "mayo", "hunyo", "hulyo", "agosto", "setyembre|septiyembre",
    "oktubre", "nobyembre", "disyembre", "lunes", "martes", "miyerkules", "huwebes", "biyernes",
    "sabado", "linggo", "ngayon", "bukas", "kahapon", "ama|tatay", "ina|nanay", "anak na lalaki",
    "anak na babae", "lolo", "lola", "tiyuhin|tito", "tiyahin|tita|aunt", "pinsan", "mga magulang",
    "batang lalaki", "batang babae", "lalaki", "babae", "bingi", "mahina ang pandinig|hirap makarinig",
    "gumagamit ng wheelchair|wheelchair user|wheelchair person", "bulag", "bingi at bulag|deafblind",
    "may asawa|kasal", "asul|bughaw", "berde|luntian", "pula", "kayumanggi|kulay kape", "itim", "puti",
    "dilaw", "kahel|kulay dalandan", "abo|abuhin|grey", "rosas|kulay rosas", "lila|ube|purple",
    "mapusyaw|light color|light colour", "madilim|madilim na kulay|dark color|dark colour", "tinapay", "itlog", "isda", "karne", "manok",
    "ispageti", "kanin|bigas", "longganisa", "hipon", "alimango", "mainit", "malamig", "katas|dyus",
    "gatas", "kape", "tsaa", "serbesa", "alak", "asukal", "walang asukal|no added sugar",
]


def bundle(source: Path, workers: int = 4) -> list[dict]:
    with (source / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    ids = [int(row["id"]) for row in labels]
    if len(ids) != 105 or set(ids) != set(range(105)) or len(TRANSLATIONS) != 105:
        raise ValueError("Expected the original 105 unique model classes")
    output = ROOT / "assets" / "lessons"
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def convert(row: dict) -> dict:
        class_id = int(row["id"])
        folder = source / "clips" / str(class_id)
        candidates = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".mov", ".mp4"})
        preferred = [p for p in candidates if p.stem == "0"]
        if not candidates:
            raise ValueError(f"No demonstration for class {class_id}")
        original = (preferred or candidates)[0]
        target = output / f"{class_id:03d}.mp4"
        subprocess.run([
            ffmpeg, "-y", "-v", "error", "-i", str(original), "-map", "0:v:0",
            "-an", "-sn", "-dn", "-map_metadata", "-1", "-c:v", "libx264",
            "-threads", "1", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target),
        ], check=True, capture_output=True)
        inspection = subprocess.run([ffmpeg, "-i", str(target)], capture_output=True, text=True).stderr
        if "Audio:" in inspection or "Video: h264" not in inspection:
            raise ValueError(f"Unexpected media streams in {target}")
        subprocess.run([ffmpeg, "-v", "error", "-i", str(target), "-f", "null", "-"],
                       check=True, capture_output=True)
        return {
            "id": class_id, "label": row["label"].strip(), "category": row["category"].strip(),
            "video": f"assets/lessons/{target.name}", "source_clip": original.relative_to(source).as_posix(),
            "audio_tracks": 0, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        videos = sorted(executor.map(convert, labels), key=lambda row: row["id"])
    manifest = {"source_labels": "config/labels.csv", "videos": videos}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copyfile(source / "labels.csv", ROOT / "config" / "labels.csv")
    corrections = {5: "I'M FINE", 8: "YOU'RE WELCOME", 68: "WHEELCHAIR USER"}
    with (ROOT / "config" / "lessons.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "id", "label", "category", "filipino", "synonyms", "beginner_order", "display_label", "video"
        ])
        writer.writeheader()
        for row in sorted(labels, key=lambda row: int(row["id"])):
            class_id = int(row["id"])
            equivalents = TRANSLATIONS[class_id].split("|")
            display = corrections.get(class_id, row["label"].strip())
            writer.writerow({
                "id": class_id, "label": row["label"].strip(), "category": row["category"].strip(),
                "filipino": equivalents[0], "synonyms": "|".join([display, *equivalents]),
                "beginner_order": class_id + 1, "display_label": display,
                "video": f"assets/lessons/{class_id:03d}.mp4",
            })
    return videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    videos = bundle(args.dataset)
    print(json.dumps({"bundled_videos": len(videos), "audio_tracks": 0,
                      "megabytes": round(sum(v["bytes"] for v in videos) / 1024 ** 2, 2)}))
