#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Етап 5 ІНДЗ: ПРОБНЕ застосування текстової моделі та незалежна контрольна вибірка.

Встановлення: python -m pip install scikit-learn joblib
Запуск у корені проєкту: python telegram_stage5_screen_and_audit.py

Використовує лише ваші локальні файли попередніх етапів. Реакції НЕ є входом
класифікатора і НЕ використовуються при формуванні анонімної контрольної вибірки.

ЗВЕРНІТЬ УВАГУ: оцінки моделі НЕ є доведеними вимірами психологічного впливу.
Мітки класів лише описують форму тексту. Модель навчена на 75 документах,
її точність на великому корпусі ще НЕ перевірена.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys

# Import exact preprocessing from stage3, to avoid input mismatch.
try:
    from telegram_stage3_annotation import tidy_for_label, week_of_month
except ImportError as exc:
    raise SystemExit("Поклади telegram_stage3_annotation.py поруч із цим скриптом.") from exc

BASE = Path(__file__).resolve().parent
PREPARED = BASE / 'results_september_stage2' / 'posts_prepared.jsonl'
MODEL = BASE / 'results_september_stage4' / 'stage4_binary_pilot_model.joblib'
ANNOT_KEY = BASE / 'results_september_stage3' / 'annotation_key_DO_NOT_OPEN_BEFORE_LABELING.csv'
OUTPUT = BASE / 'results_september_stage5'


def read_csv(path: Path):
    if not path.is_file():
        raise SystemExit(f"Не знайдено файл: {path}")
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, columns, records):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def read_rows(path: Path, ignored_ids: set[int], min_len: int, max_len: int):
    if not path.is_file():
        raise SystemExit(f"Не знайдено дані: {path}. Спочатку виконай другий етап.")
    stats = Counter()
    records = []
    seen = set()
    with path.open('r', encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f'Неправильний JSON на рядку {lineno}: {exc}') from exc
            stats['prepared_rows_read'] += 1
            if not isinstance(r, dict):
                stats['invalid_rows'] += 1
                continue
            post_row = r.get('post_row')
            if not isinstance(post_row, int) or post_row in seen:
                stats['invalid_or_repeated_id'] += 1
                continue
            seen.add(post_row)
            # Keep exactly same eligibility requirements as Stage 3.
            if r.get('same_channel_day_exact_duplicate'):
                stats['duplicate_flag_excluded'] += 1
                continue
            channel, day = r.get('channel'), r.get('date_utc')
            if not isinstance(channel, str) or not channel or not week_of_month(day):
                stats['bad_channel_or_day'] += 1
                continue
            text = tidy_for_label(r.get('text', ''))
            if len(text) < min_len:
                stats['too_short'] += 1
                continue
            if len(text) > max_len:
                stats['too_long'] += 1
                continue
            records.append({
                'post_row': post_row, 'date_utc': day, 'channel': channel,
                'week': week_of_month(day), 'text': text,
                'reaction_count': int(r.get('reaction_count') or 0),
                'comments_count': int(r.get('comments_count') or 0),
                'in_stage3_pilot': post_row in ignored_ids,
            })
    stats['in_scope_for_screening'] = len(records)
    stats['previously_annotated_in_scope'] = sum(r['in_stage3_pilot'] for r in records)
    return records, stats


def sample_audit(records: list[dict], n: int, seed: int):
    """Select from 3 SCORE QUANTILES; no reactions or channel metadata shown to annotator.

    This purposely enriches the uncertain/likely emotional cases, so audit metrics
    cannot be interpreted as the distribution of real-world Telegram traffic.
    """
    pool = [r for r in records if not r['in_stage3_pilot']]
    if len(pool) < n:
        raise SystemExit('Недостатньо нових публікацій для незалежної контрольної вибірки.')
    pool.sort(key=lambda r: (r['screen_score'], r['post_row']))
    # Equal-count empirical thirds (rank groups, not assumed calibrated probabilities).
    k = len(pool)
    strata = [
        ('lower_third', pool[:k // 3]),
        ('middle_third', pool[k // 3: (2 * k) // 3]),
        ('upper_third', pool[(2 * k) // 3:]),
    ]
    rng = random.Random(seed)
    desired = [n // 3] * 3
    for i in range(n % 3):
        desired[i] += 1
    chosen = []
    used = set()
    for (name, group), quota in zip(strata, desired):
        # Within each score group, choose across channels first to avoid a few
        # very large channels dominating the enriched validation set.
        by_channel = defaultdict(list)
        for r in group:
            by_channel[r['channel']].append(r)
        channels = list(by_channel)
        rng.shuffle(channels)
        for ch in channels:
            rng.shuffle(by_channel[ch])
        selected = []
        while len(selected) < quota:
            advanced = False
            for ch in channels:
                while by_channel[ch] and by_channel[ch][-1]['post_row'] in used:
                    by_channel[ch].pop()
                if not by_channel[ch]:
                    continue
                r = by_channel[ch].pop()
                used.add(r['post_row'])
                selected.append((name, r))
                advanced = True
                if len(selected) == quota:
                    break
            if not advanced:
                raise SystemExit(f'Не вдалося сформувати страту {name}')
        chosen.extend(selected)
    rng.shuffle(chosen)
    audit = []
    key = []
    for i, (stratum, r) in enumerate(chosen, 1):
        a_id = f'B{i:04d}'
        audit.append({
            'annotation_id': a_id, 'text_for_label': r['text'],
            'manual_emotion': '', 'manual_intensity': '',
            'manual_rhetorical_cues': '', 'manual_topic': '',
            'manual_uncertain': '', 'manual_notes': '',
        })
        key.append({
            'annotation_id': a_id, 'post_row': r['post_row'],
            'channel': r['channel'], 'date_utc': r['date_utc'],
            'score_stratum': stratum, 'predicted_binary': r['predicted_binary'],
            'model_score_UNCALIBRATED': round(r['screen_score'], 5),
            'reaction_count': r['reaction_count'],
            'comments_count': r['comments_count'],
        })
    return audit, key, {name: {
        'count_in_population': len(group),
        'score_min': round(min(x['screen_score'] for x in group), 5),
        'score_max': round(max(x['screen_score'] for x in group), 5),
        'selected': quota,
    } for (name, group), quota in zip(strata, desired)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PREPARED)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--key', type=Path, default=ANNOT_KEY)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--audit-size', type=int, default=90)
    parser.add_argument('--seed', type=int, default=2027)
    args = parser.parse_args()
    if not 15 <= args.audit_size <= 600:
        parser.error('--audit-size має бути від 15 до 600')
    if not args.model.is_file():
        raise SystemExit(f'Не знайдено модель етапу 4: {args.model}')

    # joblib uses pickle internally. Load ONLY your own model from Stage 4.
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit('Встанови: python -m pip install scikit-learn joblib') from exc
    model_bundle = joblib.load(args.model)
    if not isinstance(model_bundle, dict) or model_bundle.get('purpose') != 'preliminary text-only binary classifier':
        raise SystemExit('Неправильний формат моделі. Потрібна stage4_binary_pilot_model.joblib.')
    clf = model_bundle.get('pipeline')
    if not hasattr(clf, 'predict_proba'):
        raise SystemExit('Модель не підтримує оцінку predict_proba. На етапі 4 обери tfidf_logreg.')

    annotations = read_csv(args.key)
    if not annotations or not {'post_row', 'annotation_id'}.issubset(annotations[0]):
        raise SystemExit('Некоректний stage3 annotation key.')
    previous = {int(x['post_row']) for x in annotations}
    rows, filters = read_rows(args.input, previous, min_len=40, max_len=1400)
    if not rows:
        raise SystemExit('Немає придатних повідомлень. Перевір вхідний файл.')
    print(f'Опрацьовую {len(rows)} текстів пакетами...', flush=True)

    # Batch inference; does not copy 35M reactions / 700k comments to RAM.
    batch_size = 512
    for i in range(0, len(rows), batch_size):
        block = rows[i:i + batch_size]
        scores = clf.predict_proba([r['text'] for r in block])
        classes = list(clf.classes_)
        if 1 not in classes:
            raise SystemExit('Модель не має емоційного класу 1.')
        positive_idx = classes.index(1)
        for r, score in zip(block, scores):
            r['screen_score'] = float(score[positive_idx])
            r['predicted_binary'] = 'emotional' if r['screen_score'] >= 0.5 else 'neutral'

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    audit, audit_key, strata_stats = sample_audit(rows, args.audit_size, args.seed)
    save_csv(out / 'stage5_blind_audit.csv', list(audit[0]), audit)
    save_csv(out / 'stage5_audit_key_DO_NOT_OPEN_BEFORE_LABELING.csv', list(audit_key[0]), audit_key)

    channel_stats = defaultdict(list)
    day_stats = defaultdict(list)
    channel_day_stats = defaultdict(list)
    predictions = []
    for r in rows:
        channel_stats[r['channel']].append(r['screen_score'])
        day_stats[r['date_utc']].append(r['screen_score'])
        channel_day_stats[(r['channel'], r['date_utc'])].append(r['screen_score'])
        predictions.append({
            'post_row': r['post_row'], 'date_utc': r['date_utc'], 'channel': r['channel'],
            'predicted_binary_PILOT': r['predicted_binary'],
            'model_score_UNCALIBRATED': f"{r['screen_score']:.6f}",
            'in_stage3_label_sample': 'yes' if r['in_stage3_pilot'] else 'no',
            'text_for_model': r['text'],
        })
    save_csv(out / 'stage5_pilot_predictions.csv', list(predictions[0]), predictions)

    def stats_line(scores):
        return {
            'n_posts': len(scores),
            'mean_model_score_UNCALIBRATED': round(statistics.mean(scores), 5),
            'predicted_emotional_share_PILOT': round(sum(s >= 0.5 for s in scores) / len(scores), 5),
        }

    channels = [{'channel': name, **stats_line(s)} for name, s in sorted(channel_stats.items())]
    days = [{'date_utc': day, **stats_line(s)} for day, s in sorted(day_stats.items())]
    channel_days = [{'channel': ch, 'date_utc': day, **stats_line(s)}
                    for (ch, day), s in sorted(channel_day_stats.items())]
    save_csv(out / 'stage5_channels_PILOT.csv', list(channels[0]), channels)
    save_csv(out / 'stage5_days_PILOT.csv', list(days[0]), days)
    save_csv(out / 'stage5_channel_days_PILOT.csv', list(channel_days[0]), channel_days)

    summary = {
        'stage': 5,
        'objective': 'EXPLORATORY screening of textual emotional framing; NEW blinded enriched evaluation sample',
        'input': str(args.input),
        'model': str(args.model),
        'model_features': 'text only: stage3 tidy_for_label() + stage4 TF-IDF',
        'sampling_excludes_stage3_100_from_audit': True,
        'filters': dict(filters),
        'corpus_screening': stats_line([r['screen_score'] for r in rows]),
        'new_audit_sample_size': len(audit),
        'new_audit_sample_strata': strata_stats,
        'new_audit_unique_channels': len({k['channel'] for k in audit_key}),
        'new_audit_week_counts': dict(sorted(Counter(week_of_month(k['date_utc']) for k in audit_key).items())),
        'outputs': ['stage5_blind_audit.csv', 'stage5_audit_key_DO_NOT_OPEN_BEFORE_LABELING.csv',
                    'stage5_pilot_predictions.csv', 'stage5_channels_PILOT.csv',
                    'stage5_days_PILOT.csv', 'stage5_channel_days_PILOT.csv'],
        'interpretation_warnings': [
            'Pilot model fitted on only 75 reviewed-but-not-independently-annotated texts.',
            'Logistic regression predict_proba is NOT empirically calibrated in this pilot.',
            'Do NOT present predicted emotional share as a measured fraction of emotional posts.',
            'Audit is enriched by model score thirds and channel-balanced; do NOT treat it as prevalence-representative.',
            'The previous 100 texts are excluded from audit but may be included in descriptive model scores.',
            'No reaction/engagement data were features; no exposure, timing or causation can be inferred.',
            'Do not evaluate the new audit until human labels are completed BLIND to the key.',
        ],
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    (out / 'stage5_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'README_STAGE5_UA.txt').write_text(
        'ЕТАП 5: ПІЛОТНЕ ОЦІНЮВАННЯ ТЕКСТІВ + НОВА СЛІПА ПЕРЕВІРКА\n'
        '1. stage5_summary.json: підсумок, надішли асистенту.\n'
        '2. stage5_blind_audit.csv: нові тексти для розмітки без прогнозу / реакцій.\n'
        '   Почни з 20-30 рядків, пізніше заповни всі.\n'
        '3. stage5_audit_key_DO_NOT_OPEN_BEFORE_LABELING.csv: НЕ ВІДКРИВАЙ до розмітки.\n'
        '4. stage5_pilot_predictions.csv: попередні прогнози, НЕ підсумкові факти.\n'
        '5. stage5_channels_PILOT.csv, stage5_days_PILOT.csv,\n'
        '   stage5_channel_days_PILOT.csv: описові агрегати, НЕ психологічний вплив.\n'
        'Ніяких висновків про реальні поширення, вплив на людей чи причинність.\n',
        encoding='utf-8'
    )
    print('Успішно! Результати:', out.resolve())
    print('Вибірка для нового сліпого маркування: ', len(audit))
    print('Надішли stage5_summary.json і після розмітки stage5_blind_audit.csv.')
    print('НЕ відкривай файл ключа до завершення нової розмітки.')


if __name__ == '__main__':
    main()
