"""Stage 1: audit ONLY the September 2026 Aisberg Telegram monthly file.

Requirements: Python >=3.9, `pip install huggingface_hub`.

Run:
    python telegram_stage1_inspect.py


Produces:
  results_september/data_audit.json     Complete statistics (safe compact report)
  results_september/sample_posts.jsonl   Up to three posts per channel for manual inspection
  results_september/schema_example.json  Observed field types, no post text

No causal inference / diffusion claims are made at this stage.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

REPO = "aisbergpublicorganization/telegram-news-ua-dataset"
FILENAME = "data/09_2026.jsonl"
EXPECTED_FIELDS = {"channel", "post_text_redacted", "reactions", "comments", "date"}


def fetch_dataset():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install dependency: python -m pip install huggingface_hub") from exc
    print("Завантажую вересень 2026 з Hugging Face (файл близько 147 МБ)...")
    return Path(hf_hub_download(repo_id=REPO, filename=FILENAME, repo_type="dataset"))


def type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def parse_date(value):
    # Date is documented as epoch milliseconds truncated to midnight UTC.
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def inspect(input_path, output_dir, sample_per_channel=3):
    output_dir.mkdir(exist_ok=True, parents=True)
    counts = Counter()
    by_channel, by_day, emojis = Counter(), Counter(), Counter()
    field_types = defaultdict(Counter)
    comment_field_types = defaultdict(Counter)
    commenters = set()
    day_comment_counts = Counter()
    sample_by_channel = defaultdict(list)
    duplicate_keys = set()
    sha = sha256()
    bytes_read = 0

    with input_path.open("rb") as fh:
        for line_no, raw in enumerate(fh, 1):
            sha.update(raw)
            bytes_read += len(raw)
            if not raw.strip():
                counts["empty_lines"] += 1
                continue
            try:
                record = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                counts["invalid_json_lines"] += 1
                continue
            if not isinstance(record, dict):
                counts["non_object_lines"] += 1
                continue
            counts["posts"] += 1
            for key, value in record.items():
                field_types[key][type_name(value)] += 1
            for field in EXPECTED_FIELDS - record.keys():
                counts[f"missing_field__{field}"] += 1

            channel = record.get("channel")
            if not isinstance(channel, str) or not channel.strip():
                counts["posts_without_channel"] += 1
                channel = "[UNKNOWN]"
            else:
                channel = channel.strip()
            by_channel[channel] += 1

            post = record.get("post_text_redacted")
            if not isinstance(post, str) or not post.strip():
                counts["posts_without_text"] += 1
                post = ""
            else:
                counts["posts_with_text"] += 1

            dt = parse_date(record.get("date"))
            if dt is None:
                counts["posts_invalid_date"] += 1
                date = None
            else:
                date = dt.strftime("%Y-%m-%d")
                by_day[date] += 1
                if (dt.year, dt.month) != (2026, 9):
                    counts["posts_not_in_september"] += 1
                if (dt.hour, dt.minute, dt.second, dt.microsecond) != (0, 0, 0, 0):
                    counts["posts_non_midnight_utc"] += 1

            duplicate_signature = sha256((channel + "\0" + str(record.get("date")) + "\0" + post).encode("utf-8")).digest()
            if duplicate_signature in duplicate_keys:
                counts["potential_duplicate_posts"] += 1
            else:
                duplicate_keys.add(duplicate_signature)

            reactions = record.get("reactions")
            if isinstance(reactions, dict):
                if reactions:
                    counts["posts_with_reactions"] += 1
                for emoji, amount in reactions.items():
                    if isinstance(amount, int) and not isinstance(amount, bool) and amount >= 0:
                        emojis[str(emoji)] += amount
                        counts["reaction_events"] += amount
                    else:
                        counts["invalid_reaction_values"] += 1
            else:
                counts["reactions_not_object"] += 1

            comments = record.get("comments")
            if isinstance(comments, list):
                if comments:
                    counts["posts_with_comments"] += 1
                counts["comments"] += len(comments)
                if date:
                    day_comment_counts[date] += len(comments)
                for comment in comments:
                    if not isinstance(comment, dict):
                        counts["comments_not_objects"] += 1
                        continue
                    for key, value in comment.items():
                        comment_field_types[key][type_name(value)] += 1
                    if isinstance(comment.get("user_name"), str) and comment["user_name"].strip():
                        commenters.add(comment["user_name"])
                    if not isinstance(comment.get("text"), str):
                        counts["comments_without_text"] += 1
                    if parse_date(comment.get("time")) is None:
                        counts["comments_invalid_date"] += 1
            else:
                counts["comments_not_list"] += 1

            # Small excerpt only; intentionally omit commenter pseudonyms.
            if len(sample_by_channel[channel]) < sample_per_channel:
                comment_samples = []
                if isinstance(comments, list):
                    comment_samples = [c.get("text", "")[:250] for c in comments if isinstance(c, dict) and isinstance(c.get("text"), str)][:2]
                sample_by_channel[channel].append({
                    "channel": channel,
                    "date_utc": date,
                    "post_text_excerpt": post[:500],
                    "reactions": reactions if isinstance(reactions, dict) else None,
                    "comment_count": len(comments) if isinstance(comments, list) else None,
                    "comment_text_excerpts": comment_samples,
                })

    report = {
        "dataset": REPO,
        "source_filename": FILENAME,
        "local_input_filename": input_path.name,
        "file_size_bytes": bytes_read,
        "file_sha256": sha.hexdigest(),
        "observed_counts": dict(counts),
        "number_of_channels": len(by_channel),
        "posts_by_channel": dict(by_channel.most_common()),
        "posts_by_day_utc": dict(sorted(by_day.items())),
        "comments_by_day_utc": dict(sorted(day_comment_counts.items())),
        "reactions_by_emoji": dict(emojis.most_common()),
        "unique_commenter_pseudonyms": len(commenters),
        "audit_notes": [
            "The monthly file has daily timestamps, not minute-level event times.",
            "The monthly file does not supply an observed repost or follower graph.",
            "Sentiment/emotion and emoji counts are proxies, not proof of psychological causation.",
            "Reported comment pseudonym count may not equal distinct real users.",
        ],
    }
    schema = {
        "post_field_observed_types_and_counts": {k: dict(v) for k, v in sorted(field_types.items())},
        "comment_field_observed_types_and_counts": {k: dict(v) for k, v in sorted(comment_field_types.items())},
    }

    report_path = output_dir / "data_audit.json"
    schema_path = output_dir / "schema_example.json"
    samples_path = output_dir / "sample_posts.jsonl"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    with samples_path.open("w", encoding="utf-8") as fh:
        for channel, posts in sorted(sample_by_channel.items()):
            for post in posts:
                fh.write(json.dumps(post, ensure_ascii=False) + "\n")

    print(f"Всього постів: {counts['posts']:,}; каналів: {len(by_channel)}; коментарів: {counts['comments']:,}")
    print("Звіти збережено до:", output_dir.resolve())
    for path in (report_path, schema_path, samples_path):
        print(" -", path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="Existing downloaded 09_2026.jsonl file")
    parser.add_argument("--out", type=Path, default=Path("results_september"))
    parser.add_argument("--sample-per-channel", type=int, default=3)
    args = parser.parse_args()
    if args.sample_per_channel < 0 or args.sample_per_channel > 20:
        parser.error("--sample-per-channel must be between 0 and 20")
    path = args.file if args.file is not None else fetch_dataset()
    if not path.is_file():
        parser.error(f"Файл не знайдено: {path}")
    inspect(path, args.out, args.sample_per_channel)

if __name__ == "__main__":
    main()
