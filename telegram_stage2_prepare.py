#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2. September 2026 Telegram corpus: safe preprocessing and exploratory features.

Uses Python 3.9+ standard library only, plus huggingface_hub for automatic download.
Do not interpret emoji categories as measurements of an individual's psychological state.
Does not infer actual reposts, influence direction, or sentiment from text.

Run:
    py telegram_stage2_prepare.py --file "C:/path/to/09_2026.jsonl"
Or (automatically fetch the exact September snapshot into Hugging Face cache):
    py -m pip install huggingface_hub
    py telegram_stage2_prepare.py

Outputs in results_september_stage2/: stage2_summary.json, channel_summary.csv,
posts_prepared.jsonl, annotation_sample.csv, cross_channel_exact_examples.json.
Send only stage2_summary.json and channel_summary.csv to the assistant for next step.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import re

REPO = "aisbergpublicorganization/telegram-news-ua-dataset"
FILENAME = "data/09_2026.jsonl"
EXPECTED_SHA256 = "bd75dfb5e3cc652248bf69c9b492ba3294893db69d5feaf3da3c477a889a6608"

# These are ONLY coarse counts of explicit emoji choices, NOT sentiment diagnoses.
EMOJI_GROUPS = {
    "anger": {"🤬", "😡"},
    "sadness": {"😢", "😭", "💔"},
    "fear": {"😱", "😨"},
    "laughter": {"😁", "🤣", "😂"},
}

WHITESPACE = re.compile(r"\s+")
MARKDOWN_URL = re.compile(r"\[([^\]]+)\]\(\s*(?:\[url\]|https?://[^)]*)\s*\)", re.I)
PLACEHOLDER = re.compile(r"\[(?:url|mention|email|phone|id)\]", re.I)


def utc_day(value):
    """Dates in this dataset are Unix milliseconds rounded to UTC midnight."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return None


def clean_text(value):
    if not isinstance(value, str):
        return ""
    # Keep the visible text of complete Markdown links, remove anonymized URLs;
    # malformed links from redaction may leave punctuation -- tolerate it.
    s = MARKDOWN_URL.sub(r"\1", value)
    s = PLACEHOLDER.sub(" ", s)
    s = s.replace("**", "").replace("__", "").replace("~~", "")
    return WHITESPACE.sub(" ", s).strip()


def counts_by_emoji(obj):
    if not isinstance(obj, dict):
        return Counter()
    return Counter({str(key): val for key, val in obj.items()
                    if isinstance(val, int) and not isinstance(val, bool) and val > 0})


def limited_mean(total, n):
    return round(total / n, 3) if n else None


def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise SystemExit("Install: py -m pip install huggingface_hub") from err
    return Path(hf_hub_download(repo_id=REPO, filename=FILENAME, repo_type="dataset"))


def process(file_path, output_dir, sample_per_channel=3, min_chars=25,
            seed=2026, verify_sha=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_file = output_dir / "posts_prepared.jsonl"

    quality = Counter()
    channel_stats = defaultdict(Counter)
    group_totals = Counter()
    day_stats = defaultdict(Counter)
    seen_same_channel = set()
    crosschannel_map = defaultdict(set)  # nonempty exact text + day => channel names
    crosschannel_example = {}  # bounded snippets for audit
    samples = defaultdict(list)
    candidate_count_by_channel = Counter()
    rng = random.Random(seed)
    hasher = sha256()

    with file_path.open("rb") as source, prepared_file.open("w", encoding="utf-8") as out:
        for row, raw in enumerate(source, start=1):
            hasher.update(raw)
            if not raw.strip():
                quality["empty_line"] += 1
                continue
            try:
                post = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                quality["invalid_json"] += 1
                continue
            if not isinstance(post, dict):
                quality["not_dict"] += 1
                continue
            quality["all_posts"] += 1
            channel = post.get("channel")
            channel = channel.strip() if isinstance(channel, str) else ""
            if not channel:
                quality["missing_channel"] += 1
                channel = "[UNKNOWN]"
            stats = channel_stats[channel]
            stats["posts"] += 1
            day = utc_day(post.get("date"))
            if day is None or not day.startswith("2026-09-"):
                quality["invalid_or_outside_month"] += 1
                continue
            day_stats[day]["posts"] += 1

            raw_text = post.get("post_text_redacted", "")
            txt = clean_text(raw_text)
            if not txt:
                quality["no_clean_text"] += 1
                stats["no_clean_text"] += 1
            else:
                quality["with_clean_text"] += 1
                stats["with_clean_text"] += 1
            if len(txt) < min_chars:
                quality["below_min_chars"] += 1
                stats["below_min_chars"] += 1

            reactions = counts_by_emoji(post.get("reactions"))
            reactions_total = sum(reactions.values())
            stats["reactions_total"] += reactions_total
            day_stats[day]["reactions"] += reactions_total
            if reactions_total:
                stats["posts_with_reactions"] += 1
                quality["with_reactions"] += 1
            group_n = {g: sum(reactions[e] for e in emojis) for g, emojis in EMOJI_GROUPS.items()}
            for group, value in group_n.items():
                group_totals[group] += value
                stats[group + "_reactions"] += value
            unclassified = reactions_total - sum(group_n.values())
            group_totals["other_or_ambiguous"] += unclassified
            comments = post.get("comments")
            comments = comments if isinstance(comments, list) else []
            stats["comments_total"] += len(comments)
            day_stats[day]["comments"] += len(comments)
            if comments:
                stats["posts_with_comments"] += 1
                quality["posts_with_comments"] += 1
            # No usernames, user-level features or full comment texts in outputs.
            stats["nonempty_comments"] += sum(1 for c in comments
                                               if isinstance(c, dict) and isinstance(c.get("text"), str)
                                               and c["text"].strip())
            quality["comments_total"] += len(comments)

            is_duplicate = False
            if len(txt) >= min_chars:
                canonical = txt.casefold()
                content_hash = sha256(canonical.encode("utf-8")).hexdigest()
                same_key = (channel, day, content_hash)
                is_duplicate = same_key in seen_same_channel
                if is_duplicate:
                    quality["exact_duplicate_text_same_channel_day"] += 1
                    stats["exact_duplicate_text_same_channel_day"] += 1
                seen_same_channel.add(same_key)

                cross_key = (day, content_hash)
                crosschannel_map[cross_key].add(channel)
                if cross_key not in crosschannel_example:
                    crosschannel_example[cross_key] = txt[:180]
            else:
                content_hash = None

            if len(txt) < min_chars:
                continue  # source data unchanged; excluded ONLY from text model export
            quality["prepared_posts"] += 1
            stats["prepared_posts"] += 1
            result = {
                "post_row": row,
                "date_utc": day,
                "channel": channel,
                "text": txt,
                "text_sha256": content_hash,
                "char_count": len(txt),
                "same_channel_day_exact_duplicate": is_duplicate,
                "reaction_count": reactions_total,
                "comments_count": len(comments),
                # Shares use count of emoji reactions as denominator, not population size.
                "emoji_anger_share": limited_mean(group_n["anger"], reactions_total),
                "emoji_sadness_share": limited_mean(group_n["sadness"], reactions_total),
                "emoji_fear_share": limited_mean(group_n["fear"], reactions_total),
                "emoji_laughter_share": limited_mean(group_n["laughter"], reactions_total),
                "emoji_other_share": limited_mean(unclassified, reactions_total),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Deterministic reservoir sampling per channel for HUMAN review.
            candidate_count_by_channel[channel] += 1
            s = samples[channel]
            if len(s) < sample_per_channel:
                s.append(result)
            elif sample_per_channel:
                index = rng.randrange(candidate_count_by_channel[channel])
                if index < sample_per_channel:
                    s[index] = result

    actual_digest = hasher.hexdigest()
    if verify_sha and actual_digest != EXPECTED_SHA256:
        raise RuntimeError("SHA-256 mismatch: local file is not the audited September snapshot")

    ambiguous_groups = sorted({channel for channel in channel_stats if
                              any(other != channel and (channel.lower() in other.lower()
                                                        or other.lower() in channel.lower())
                                  for other in channel_stats)})
    repeated_cross = [(day, channels, crosschannel_example[(day, key)])
                      for (day, key), channels in crosschannel_map.items() if len(channels) > 1]
    repeated_cross.sort(key=lambda x: (-len(x[1]), x[0]))
    cross_examples = [{"day_utc": day, "channels": sorted(channels), "text_excerpt": snippet}
                      for day, channels, snippet in repeated_cross[:15]]
    (output_dir / "cross_channel_exact_examples.json").write_text(
        json.dumps(cross_examples, ensure_ascii=False, indent=2), encoding="utf-8")

    total_reactions = sum(c["reactions_total"] for c in channel_stats.values())
    summary = {
        "source_file": str(file_path),
        "file_sha256": actual_digest,
        "matches_audited_sha256": actual_digest == EXPECTED_SHA256,
        "parameters": {"min_chars": min_chars, "sample_per_channel": sample_per_channel,
                       "seed": seed, "time_resolution": "UTC day only"},
        "quality": dict(quality),
        "channel_name_count": len(channel_stats),
        "possible_channel_name_aliases_for_manual_review": ambiguous_groups,
        "cross_channel_exact_text_day_signatures": len(repeated_cross),
        "emoji_categories_count": dict(group_totals),
        "emoji_categories_share_of_all_reactions": {
            group: limited_mean(count, total_reactions) for group, count in group_totals.items()
        },
        "daily_stats": {k: dict(v) for k, v in sorted(day_stats.items())},
        "limitations": [
            "No observed repost/follower edges; similar posts are NOT evidence of a causal channel-to-channel flow.",
            "Only day-level UTC timing; same-day posts cannot be temporally ordered.",
            "Emoji shares describe reaction events, not unique people or validated emotions.",
            "No exposures/views/subscriber denominators in this file.",
            "Channel name aliases are NOT merged automatically.",
            "Same-channel identical text duplicates are flagged but NOT removed automatically.",
            "Text-only preprocessing excludes blank/media-only posts from model input, not from raw audit.",
            "Clean text may contain source branding or damaged Markdown artifacts; review samples.",
        ],
    }
    (output_dir / "stage2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "channel_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        columns = ["channel", "posts", "prepared_posts", "no_clean_text",
                   "exact_duplicate_text_same_channel_day", "reactions_total",
                   "comments_total", "posts_with_reactions", "posts_with_comments",
                   "anger_reaction_share", "sadness_reaction_share", "fear_reaction_share",
                   "laughter_reaction_share"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for channel in sorted(channel_stats):
            s = channel_stats[channel]
            writer.writerow({"channel": channel, **{k: s[k] for k in columns[1:-4]},
                             "anger_reaction_share": limited_mean(s["anger_reactions"], s["reactions_total"]),
                             "sadness_reaction_share": limited_mean(s["sadness_reactions"], s["reactions_total"]),
                             "fear_reaction_share": limited_mean(s["fear_reactions"], s["reactions_total"]),
                             "laughter_reaction_share": limited_mean(s["laughter_reactions"], s["reactions_total"])})

    with (output_dir / "annotation_sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        columns = ["post_row", "date_utc", "channel", "text", "reaction_count",
                   "emoji_anger_share", "emoji_sadness_share", "emoji_fear_share",
                   "emoji_laughter_share", "comments_count", "manual_topic",
                   "manual_text_emotion", "manual_notes"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for channel in sorted(samples):
            for post in samples[channel]:
                writer.writerow({k: post.get(k, "") for k in columns})

    print(f"Прочитано: {quality['all_posts']:,} записів; підготовлено: {quality['prepared_posts']:,} текстових записів.")
    print(f"Повтори однакового тексту в тому самому каналі/дні: {quality['exact_duplicate_text_same_channel_day']:,}")
    print(f"Точні збіги тексту між каналами в той самий день: {len(repeated_cross):,} текстових сигнатур")
    print("SHA-256 збігається з першим аудитом:", actual_digest == EXPECTED_SHA256)
    print("Результати:", output_dir.resolve())
    print("ПРИКРІПИ: stage2_summary.json та channel_summary.csv (не потрібно posts_prepared.jsonl).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="Local 09_2026.jsonl (avoids downloading)")
    parser.add_argument("--out", type=Path, default=Path("results_september_stage2"))
    parser.add_argument("--sample-per-channel", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verify-sha", action="store_true", help="Abort if source is not exact first audited file")
    args = parser.parse_args()
    if not 0 <= args.sample_per_channel <= 20:
        parser.error("--sample-per-channel requires value 0...20")
    if args.min_chars < 1:
        parser.error("--min-chars requires value >=1")
    source = args.file if args.file else download()
    if not source.is_file():
        parser.error(f"Cannot find input file: {source}")
    process(source, args.out, args.sample_per_channel, args.min_chars, args.seed, args.verify_sha)


if __name__ == "__main__":
    main()
