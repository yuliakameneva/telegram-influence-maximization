#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Етап 7: побудова недирективного графа текстових збігів Telegram-каналів.

Запуск: python telegram_stage7_build_proxy_graph.py
Вимоги: scikit-learn, numpy, matplotlib (встановлені на етапі 4).
Усе обчислюється локально. Кількості реакцій і коментарів не використовуються.

ВАЖЛИВО: ребра — схожі тексти, опубліковані в один UTC-день.
Це НЕ спостережувані пересилання, НЕ граф підписників і НЕ причинний вплив.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as exc:
    raise SystemExit("Встанови: python -m pip install scikit-learn numpy matplotlib") from exc

try:
    from telegram_stage3_annotation import tidy_for_label, week_of_month
except ImportError as exc:
    raise SystemExit("Поряд зі скриптом має бути telegram_stage3_annotation.py") from exc

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results_september_stage2" / "posts_prepared.jsonl"
DEFAULT_OUTPUT = ROOT / "results_september_stage7"
WORD = re.compile(r"(?u)\b\w{2,}\b")


def save_csv(path: Path, cols: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_posts(path: Path, min_chars: int, max_chars: int):
    if not path.exists():
        raise SystemExit(f"Не знайдено {path}. Потрібен результат другого етапу.")
    by_day = defaultdict(list)
    counters = Counter()
    channels = Counter()
    seen_rows = set()
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            counters["prepared_rows"] += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Неправильний JSON у рядку {line_no}: {exc}") from exc
            rid = r.get("post_row")
            if not isinstance(rid, int) or rid in seen_rows:
                counters["bad_or_duplicate_row_ids"] += 1
                continue
            seen_rows.add(rid)
            if r.get("same_channel_day_exact_duplicate"):
                counters["duplicate_flag_excluded"] += 1
                continue
            day = r.get("date_utc")
            channel = r.get("channel")
            if not isinstance(channel, str) or not channel.strip() or week_of_month(day) == 0:
                counters["bad_channel_or_date"] += 1
                continue
            text = tidy_for_label(r.get("text"))
            if len(text) < min_chars:
                counters["too_short_after_cleanup"] += 1
                continue
            if len(text) > max_chars:
                counters["too_long_for_scope"] += 1
                continue
            word_set = set(WORD.findall(text.lower()))
            by_day[day].append({
                "post_row": rid, "channel": channel, "date_utc": day,
                "text": text, "terms": word_set, "chars": len(text),
            })
            channels[channel] += 1
            counters["in_scope"] += 1
    return by_day, channels, counters


def match_one_day(posts, similarity_threshold, min_overlap, min_shared_words, max_matches_per_post, tfidf_limit):
    """Find conservative same-day lexical similarities; per-channel-pair one-to-one matching."""
    if len(posts) < 2 or len({p["channel"] for p in posts}) < 2:
        return [], 0
    # Local, multilingual word and two-word phrase statistics.
    # Day-sized blocks avoid fitting an expensive model to the entire month at once.
    tfidf = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.85,
        max_features=tfidf_limit, sublinear_tf=True, norm="l2",
        dtype=np.float32, token_pattern=r"(?u)\b\w\w+\b",
    )
    try:
        matrix = tfidf.fit_transform([p["text"] for p in posts]).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "no terms remain" in str(exc):
            return [], 0
        raise

    candidate_by_pair = defaultdict(list)
    candidate_count = 0
    n = len(posts)
    for begin in range(0, n, 64):
        end = min(begin + 64, n)
        similarity = (matrix[begin:end] @ matrix.T).tocsr()
        for rel in range(end - begin):
            i = begin + rel
            start, stop = similarity.indptr[rel], similarity.indptr[rel + 1]
            inds = similarity.indices[start:stop]
            vals = similarity.data[start:stop]
            # Candidate cap is applied AFTER cross-channel / length filtering;
            # same-channel duplicates must not crowd out between-channel matches.
            idx = [k for k in np.flatnonzero((inds > i) & (vals >= similarity_threshold))
                   if posts[i]["channel"] != posts[int(inds[k])]["channel"]
                   and min(posts[i]["chars"], posts[int(inds[k])]["chars"]) /
                       max(posts[i]["chars"], posts[int(inds[k])]["chars"]) >= 0.35]
            if not idx:
                continue
            idx.sort(key=lambda k: (-float(vals[k]), int(inds[k])))
            idx = idx[:max_matches_per_post]
            for k in idx:
                j = int(inds[k])
                a, b = posts[i], posts[j]
                intersection = len(a["terms"] & b["terms"])
                if intersection < min_shared_words:
                    continue
                short_size = min(len(a["terms"]), len(b["terms"]))
                if short_size == 0 or intersection / short_size < min_overlap:
                    continue
                channels = tuple(sorted((a["channel"], b["channel"])))
                candidate_by_pair[channels].append((float(vals[k]), i, j))
                candidate_count += 1

    selected = []
    for pair, possible in candidate_by_pair.items():
        # Limit duplicated near-identical snippets in either channel on one day.
        # The same post may be matched separately to several *other* channels.
        used_a, used_b = set(), set()
        for score, i, j in sorted(possible, key=lambda v: (-v[0], posts[v[1]]["post_row"], posts[v[2]]["post_row"])):
            left, right = (i, j) if posts[i]["channel"] == pair[0] else (j, i)
            if left in used_a or right in used_b:
                continue
            used_a.add(left)
            used_b.add(right)
            selected.append((pair, score, posts[left], posts[right]))
    return selected, candidate_count


def build_graph(by_day, channel_posts, args):
    edge_data = defaultdict(lambda: {"scores": [], "days": set(), "examples": []})
    day_rows = []
    example_rows = []
    gross_candidates = 0
    for day_idx, (day, posts) in enumerate(sorted(by_day.items()), 1):
        matches, candidates = match_one_day(
            posts, args.min_similarity, args.min_overlap, args.min_shared_words,
            args.max_candidates_per_post, args.max_features,
        )
        gross_candidates += candidates
        for (ca, cb), score, a, b in matches:
            d = edge_data[(ca, cb)]
            d["scores"].append(score)
            d["days"].add(day)
            if len(d["examples"]) < 5:
                d["examples"].append((day, round(score, 4), a, b))
        day_rows.append({
            "date_utc": day, "in_scope_posts": len(posts),
            "channels": len({p["channel"] for p in posts}),
            "candidate_text_pairs": candidates,
            "retained_same_day_cross_channel_pairs": len(matches),
            "channel_pairs_with_matches": len({m[0] for m in matches}),
        })
        print(f"[{day_idx}/{len(by_day)}] {day}: {len(posts)} текстів, {len(matches)} зіставлень", flush=True)

    all_edges = []
    for (a, b), stat in edge_data.items():
        shared_days = sorted(stat["days"])
        n_pairs = len(stat["scores"])
        all_edges.append({
            "channel_a": a, "channel_b": b, "matched_post_pairs": n_pairs,
            "shared_days": len(shared_days),
            "mean_tfidf_similarity": round(float(np.mean(stat["scores"])), 4),
            "normalized_overlap_rate": round(n_pairs / math.sqrt(channel_posts[a] * channel_posts[b]), 6),
            "first_matched_day_utc": shared_days[0], "last_matched_day_utc": shared_days[-1],
        })
    all_edges.sort(key=lambda r: (-r["matched_post_pairs"], -r["shared_days"], r["channel_a"], r["channel_b"]))
    kept = [r for r in all_edges if r["matched_post_pairs"] >= args.min_edge_matches and r["shared_days"] >= args.min_shared_days]
    degree, weight = Counter(), Counter()
    for r in kept:
        for ch in (r["channel_a"], r["channel_b"]):
            degree[ch] += 1
            weight[ch] += r["matched_post_pairs"]
    nodes = [{
        "channel": ch, "eligible_post_count": n, "proxy_degree": degree[ch],
        "proxy_weighted_degree_matched_posts": weight[ch],
    } for ch, n in sorted(channel_posts.items(), key=lambda v: (-weight[v[0]], -v[1], v[0]))]

    # Representative post pairs from edges selected for the graph only.
    for r in kept:
        pair = (r["channel_a"], r["channel_b"])
        for day, score, a, b in edge_data[pair]["examples"]:
            example_rows.append({
                "date_utc": day, "channel_a": pair[0], "channel_b": pair[1],
                "post_row_a": a["post_row"], "post_row_b": b["post_row"],
                "tfidf_similarity": score,
                "text_a_excerpt": a["text"][:260], "text_b_excerpt": b["text"][:260],
            })
    example_rows.sort(key=lambda r: (-r["tfidf_similarity"], r["date_utc"]))
    return nodes, all_edges, kept, day_rows, example_rows[:200], gross_candidates


def plot_heatmap(channels, edges, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib missing"
    names = [r["channel"] for r in channels]
    pos = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for r in edges:
        i, j = pos[r["channel_a"]], pos[r["channel_b"]]
        matrix[i, j] = matrix[j, i] = r["matched_post_pairs"]
    fig, ax = plt.subplots(figsize=(13, 11))
    heat = ax.imshow(matrix, interpolation="nearest")
    # Strip pictographic symbols from plotting tick labels (not from data files).
    def plot_label(name):
        name = ''.join(ch for ch in name if ord(ch) < 0x1f000)
        return name if len(name) <= 27 else name[:26] + "…"
    tick = [plot_label(name) for name in names]
    ax.set_xticks(range(len(names)), tick, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), tick, fontsize=7)
    ax.set_title("Проксі-граф схожих публікацій: кількість зіставлень за UTC-день")
    ax.set_xlabel("Назва каналу")
    ax.set_ylabel("Назва каналу")
    fig.colorbar(heat, ax=ax, fraction=0.04, pad=0.03, label="Зіставлені пари текстів")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-similarity", type=float, default=0.70)
    parser.add_argument("--min-overlap", type=float, default=0.50)
    parser.add_argument("--min-shared-words", type=int, default=3)
    parser.add_argument("--max-candidates-per-post", type=int, default=35)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--min-edge-matches", type=int, default=3)
    parser.add_argument("--min-shared-days", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=1400)
    args = parser.parse_args()
    if not (0 < args.min_similarity <= 1 and 0 < args.min_overlap <= 1):
        parser.error("Схожість і перекриття мають бути між 0 і 1")
    if min(args.min_shared_words, args.max_candidates_per_post, args.max_features,
           args.min_edge_matches, args.min_shared_days) < 1:
        parser.error("Кількісні пороги мають бути додатними")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        parser.error("Некоректні обмеження довжини тексту")

    by_day, channel_counts, counters = read_posts(args.input, args.min_chars, args.max_chars)
    if not by_day:
        raise SystemExit("Немає придатних повідомлень")
    print(f"Текстів: {counters['in_scope']}; назв каналів: {len(channel_counts)}; днів: {len(by_day)}", flush=True)
    nodes, all_edges, kept, day_rows, examples, gross_candidates = build_graph(by_day, channel_counts, args)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    edge_cols = ["channel_a", "channel_b", "matched_post_pairs", "shared_days", "mean_tfidf_similarity",
                 "normalized_overlap_rate", "first_matched_day_utc", "last_matched_day_utc"]
    save_csv(out / "stage7_proxy_edges.csv", edge_cols, kept)
    save_csv(out / "stage7_all_channel_pairs.csv", edge_cols, all_edges)
    save_csv(out / "stage7_channel_nodes.csv", ["channel", "eligible_post_count", "proxy_degree", "proxy_weighted_degree_matched_posts"], nodes)
    save_csv(out / "stage7_daily_matches.csv", ["date_utc", "in_scope_posts", "channels", "candidate_text_pairs", "retained_same_day_cross_channel_pairs", "channel_pairs_with_matches"], day_rows)
    save_csv(out / "stage7_matched_examples.csv", ["date_utc", "channel_a", "channel_b", "post_row_a", "post_row_b", "tfidf_similarity", "text_a_excerpt", "text_b_excerpt"], examples)
    chart_status = plot_heatmap(nodes, kept, out / "stage7_graph_heatmap.png")

    # Connectivity of the descriptive graph, not an estimate of actual influence.
    adjacency = {ch: set() for ch in channel_counts}
    for e in kept:
        adjacency[e["channel_a"]].add(e["channel_b"])
        adjacency[e["channel_b"]].add(e["channel_a"])
    visited, sizes = set(), []
    for ch in adjacency:
        if ch in visited:
            continue
        stack, size = [ch], 0
        visited.add(ch)
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node] - visited:
                visited.add(neighbor)
                stack.append(neighbor)
        sizes.append(size)
    sizes.sort(reverse=True)

    summary = {
        "stage": 7,
        "objective": "UNDIRECTED descriptive channel graph from same-UTC-day near-duplicate/similar texts",
        "input": str(args.input), "input_filters": dict(sorted(counters.items())),
        "days": len(by_day), "channel_name_labels": len(channel_counts),
        "total_cross_channel_candidate_pairs_after_filters": gross_candidates,
        "total_retained_same_day_post_matches": sum(e["matched_post_pairs"] for e in all_edges),
        "channel_pairs_with_at_least_one_match": len(all_edges),
        "proxy_graph_nodes": len(nodes), "proxy_graph_edges": len(kept),
        "isolated_nodes": sum(n["proxy_degree"] == 0 for n in nodes),
        "connected_component_sizes": sizes,
        "parameters": {
            "tfidf": "daily word 1-2 grams; max_df=0.85; min_df=2; float32",
            "min_cosine_similarity": args.min_similarity,
            "min_shorter_text_word_overlap": args.min_overlap,
            "min_shared_distinct_words": args.min_shared_words,
            "max_candidates_per_post": args.max_candidates_per_post,
            "max_tfidf_features_per_day": args.max_features,
            "one_to_one_within_channel_pair_and_day": True,
            "graph_min_matched_pairs_per_channel_pair": args.min_edge_matches,
            "graph_min_distinct_days_per_channel_pair": args.min_shared_days,
            "min_clean_text_chars": args.min_chars,
            "max_clean_text_chars": args.max_chars,
        },
        "top_proxy_edges": kept[:10],
        "limitations": [
            "Channel nodes are channel LABELS; name changes/aliases might refer to the same real channel.",
            "Links are SAME-DAY TEXTUAL OVERLAP only, not observed follows, forwards, exposures or actual diffusion.",
            "A day-level timestamp cannot determine which channel published first within a day.",
            "Undirected links do not measure influence strength or transmission probability.",
            "Word TF-IDF under-detects paraphrases and cross-language equivalences; repeated templates may produce spurious matches.",
            "Selection thresholds and top candidate cap are exploratory and affect topology.",
            "No reactions, comments, emotional scores or inferred reader states are used in this graph.",
            "A later Monte Carlo diffusion experiment will test synthetic processes ON this proxy graph, not observed contagion.",
        ],
        "chart_status": "saved" if chart_status is None else chart_status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "stage7_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Етап 7. Проксі-граф тематичної подібності Telegram-каналів

**Мета.** Виявити канали, що публікували схожі тексти протягом одного UTC-дня.

- Підготовлених повідомлень, придатних для порівняння: **{counters['in_scope']}**.
- Назв каналів: **{len(channel_counts)}**; UTC-днів: **{len(by_day)}**.
- Відібрано міжканальних пар схожих повідомлень: **{summary['total_retained_same_day_post_matches']}**.
- Ребер описового графа (мінімум {args.min_edge_matches} збіги у {args.min_shared_days} різні дні): **{len(kept)}**.
- Ізольованих вузлів: **{summary['isolated_nodes']}**.

## Метод

Публікації очищено за правилами етапу 3 і розподілено за добами UTC.
Для кожної доби незалежно обчислено TF-IDF за словами та біграмами,
а косинусну схожість між різними каналами перевірено разом із лексичним
перекриттям. Усередині кожної пари каналів за день застосовано відбір
один-до-одного, щоб зменшити вплив повторів. Ребра недирективні.

## Інтерпретація

Це **мережа схожості контенту**, а не соціальний граф Telegram.
Неможливо встановити фактичні пересилання, шлях поширення, причинну
передачу або час першої публікації в межах дня. Кількість збігів не є
імовірністю впливу. Реакції та показники емоційного оформлення не
використовувалися для створення ребер.

Для подальшого алгоритмічного експерименту можна **умовно** використовувати
топологію цього графа як один зі сценаріїв синтетичних симуляцій, а не як
спостережувану мережу впливу. Порівняння має охоплювати baselines, однакові
бюджети, випадкові seeds, абляції та чутливість до параметрів графа.
"""
    (out / "stage7_method_UA.md").write_text(report, encoding="utf-8")
    print("\nЕтап 7 виконано.")
    print("Вузлів:", len(nodes), "| Ребер:", len(kept), "| Зіставлень:", summary["total_retained_same_day_post_matches"])
    print("Результати:", out.resolve())
    print("Надішли: stage7_summary.json і stage7_proxy_edges.csv")


if __name__ == "__main__":
    main()
