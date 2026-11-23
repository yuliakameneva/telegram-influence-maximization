#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 / September 2026 Telegram: blinded, stratified text annotation.

Why this stage exists:
  - Content-emotion labels must be evaluated independently of emoji responses.
  - Reactions are aggregated events, not observed emotions or causal effects.
  - The dataset has UTC-DAY timestamps and no observed repost graph.

Requirements: Python 3.9+ standard library. Nothing to pip install.

From PyCharm terminal (project root):
    python telegram_stage3_annotation.py
Or:
    python telegram_stage3_annotation.py --file results_september_stage2/posts_prepared.jsonl
    python telegram_stage3_annotation.py --sample-size 100 --seed 2026

Outputs: results_september_stage3/{stage3_summary.json,annotation_blind.csv,
annotation_key_DO_NOT_OPEN_BEFORE_LABELING.csv,annotation_guide.md,channel_day_markers.csv}
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import html
import json
from pathlib import Path
import random
import re

DEFAULT_INPUT = Path('results_september_stage2/posts_prepared.jsonl')
DEFAULT_OUT = Path('results_september_stage3')

SPACES = re.compile(r'\s+')
FRAGMENT_LINK = re.compile(r'\[([^\]\n]{1,180})\]\(\s*')
CTA_TEXT = re.compile(
    r'підпис|прислать контент|надіслати новину|з питань реклами|'
    r'підписуйся|\bfacebook\b|\byoutube\b|\btiktok\b|\bсайт\b|'
    r'^(?:радник|insider ua|труха|новина|український телеграм|'
    r'україна online|сірий кардинал|мандарин медіа|наші новини|'
    r'times of ukraine|украина 24/7|ус|око)$', re.I,
)
IBAN = re.compile(r'\bUA\d{27}\b', re.I)
EMAIL = re.compile(r'\b[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}\b')
PHONE = re.compile(r'(?<!\w)(?:\+?380[\s\-]?)?(?:\(?0\d{2}\)?[\s\-]?)\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)')
UPPER_WORD = re.compile(r'(?<!\w)[A-ZА-ЯІЇЄҐЁЭЫ]{3,}(?!\w)')
URGENCY = re.compile(r'\b(?:терміново|негайно|увага|зараз|термінов[оі]|срочно|немедленно|urgent|immediately|attention)\b', re.I)


def tidy_for_label(text: str) -> str:
    """Conservative cleanup. Keep meaningful link anchor terms, omit obvious CTA link anchors.

    This does NOT perfectly recover text from damaged Markdown, and does not attempt
    to infer what an original URL or removed identity stood for.
    """
    s = html.unescape(text if isinstance(text, str) else '')
    def replace_fragment(match: re.Match) -> str:
        caption = match.group(1).strip()
        return ' ' if CTA_TEXT.search(caption) else caption + ' '
    s = FRAGMENT_LINK.sub(replace_fragment, s)
    s = re.sub(r'\[(?:url|mention|phone|email)\]', ' ', s, flags=re.I)
    # Remove an obvious, remaining footer *only at the end*; no broad content trimming.
    s = re.sub(r'(?:\s*[|/]\s*)?(?:👈|👉)?\s*(?:Підписатись|Підписатися|Підписуйся|'
               r'Прислать контент|Надіслати новину)(?:\s+на\s+\S+)?\s*$', '', s, flags=re.I)
    s = IBAN.sub('[IBAN]', s)
    s = EMAIL.sub('[EMAIL]', s)
    s = PHONE.sub('[PHONE]', s)
    s = s.replace('**', '').replace('__', '')
    s = re.sub(r'(?:👈|👉)\s*$', '', s)
    return SPACES.sub(' ', s).strip(' |/()')


def week_of_month(d: str) -> int:
    try:
        dt = date.fromisoformat(d)
        if dt.year != 2026 or dt.month != 9:
            return 0
        return (dt.day - 1) // 7 + 1
    except (ValueError, TypeError):
        return 0


def load_prepared(path: Path, min_len: int, max_len: int):
    rows = []
    stats = Counter()
    seen = set()
    with path.open('r', encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at line {line_number}: {exc}') from exc
            stats['all_prepared_rows'] += 1
            if not isinstance(rec, dict):
                stats['non_object_rows'] += 1
                continue
            row_id = rec.get('post_row')
            if not isinstance(row_id, int) or row_id in seen:
                stats['bad_or_duplicate_post_row'] += 1
                continue
            seen.add(row_id)
            if rec.get('same_channel_day_exact_duplicate'):
                stats['flagged_same_channel_day_duplicates_excluded_from_label_sample'] += 1
                continue
            if not rec.get('channel') or week_of_month(rec.get('date_utc')) == 0:
                stats['bad_date_or_channel'] += 1
                continue
            txt = tidy_for_label(rec.get('text', ''))
            if len(txt) < min_len:
                stats['too_short_after_cleanup'] += 1
                continue
            if len(txt) > max_len:
                stats['too_long_for_pilot_annotation'] += 1
                continue
            r = {
                'post_row': row_id, 'channel': rec['channel'], 'date_utc': rec['date_utc'],
                'week': week_of_month(rec['date_utc']), 'text': txt,
                'reaction_count': rec.get('reaction_count'),
                'comments_count': rec.get('comments_count'),
                **{key: rec.get(key) for key in ('emoji_anger_share', 'emoji_sadness_share',
                                                 'emoji_fear_share', 'emoji_laughter_share',
                                                 'emoji_other_share')},
            }
            rows.append(r)
    stats['eligible_for_annotation_sampling'] = len(rows)
    return rows, stats


def stratified_sample(rows: list[dict], sample_size: int, seed: int):
    """Balanced across named channels; covers distinct weeks when possible.

    Sampling is independent of the reaction values and emotion cues.
    """
    rng = random.Random(seed)
    by_channel = defaultdict(list)
    for r in rows:
        by_channel[r['channel']].append(r)
    channels = sorted(by_channel)
    if not channels or not sample_size:
        return []
    desired_size = min(sample_size, len(rows))
    quot, rem = divmod(desired_size, len(channels))
    # Rotate extra quotas rather than preferentially favoring alphabetically first channels.
    rotated = channels[:]
    rng.shuffle(rotated)
    quotas = {c: quot + (1 if c in set(rotated[:rem]) else 0) for c in channels}
    selected = []
    selected_ids = set()
    for channel in channels:
        pool = by_channel[channel]
        weeks = defaultdict(list)
        for rec in pool:
            weeks[rec['week']].append(rec)
        week_keys = list(weeks)
        rng.shuffle(week_keys)
        target = quotas[channel]
        for week in week_keys:
            if target <= 0:
                break
            pick = rng.choice(weeks[week])
            selected.append(pick)
            selected_ids.add(pick['post_row'])
            target -= 1
        if target:
            remainder = [r for r in pool if r['post_row'] not in selected_ids]
            for pick in rng.sample(remainder, min(target, len(remainder))):
                selected.append(pick)
                selected_ids.add(pick['post_row'])
    if len(selected) < desired_size:
        remaining = [r for r in rows if r['post_row'] not in selected_ids]
        extra = rng.sample(remaining, min(desired_size - len(selected), len(remaining)))
        selected.extend(extra)
    rng.shuffle(selected)
    return selected


def channel_day_markers(rows: list[dict], out_path: Path):
    """Descriptive text punctuation signals only, NOT emotion predictions."""
    buckets = defaultdict(Counter)
    for row in rows:
        text = row['text']
        key = (row['channel'], row['date_utc'])
        b = buckets[key]
        b['n_posts'] += 1
        b['posts_with_exclamation'] += int('!' in text or '❗' in text or '‼' in text)
        b['posts_with_question_mark'] += int('?' in text or '？' in text)
        b['posts_with_urgent_lexemes'] += int(bool(URGENCY.search(text)))
        b['posts_with_caps_word'] += int(bool(UPPER_WORD.search(text)))
        rct = row.get('reaction_count')
        if isinstance(rct, (int, float)) and rct > 0:
            b['posts_with_reaction_data'] += 1
        # Important: these are observable author's surface markers, not psychological impact.
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        cols = ['channel', 'date_utc', 'n_posts', 'posts_with_exclamation',
                'posts_with_question_mark', 'posts_with_urgent_lexemes',
                'posts_with_caps_word', 'posts_with_reaction_data']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (channel, day), counter in sorted(buckets.items()):
            w.writerow({'channel': channel, 'date_utc': day,
                        **{name: counter[name] for name in cols[2:]}})
    return len(buckets)


GUIDE = '''# Інструкція для розмітки текстового емоційного забарвлення (пілот)

Мета: позначити **мовне оформлення та явно виражений афективний зміст публікації**.
Це НЕ діагностика психологічного стану читача і НЕ перевірка правдивості події.

Відкрийте `annotation_blind.csv` в Excel/LibreOffice з кодуванням UTF-8.
Заповнюйте лише стовпчики `manual_emotion`, `manual_intensity`,
`manual_rhetorical_cues`, `manual_topic`, `manual_uncertain`, `manual_notes`.
**Не дивіться** в `annotation_key_DO_NOT_OPEN_BEFORE_LABELING.csv`, поки не завершено розмітку:
там є реакції, які можуть впливати на людські судження.

## 1. manual_emotion — одна категорія
- `neutral` — стримане фактичне повідомлення без виразного емоційного оформлення
- `anger` — безпосередньо виражене обурення, агресивне оцінювання, осуд
- `sadness` — прямо виражені сум, співчуття, жалоба, горе
- `fear` — прямо виражена тривога, залякування, очікування загрози
- `hope_joy` — безпосередньо виражені радість, підтримка, надія
- `mixed` — дві чи більше чітких емоцій, без домінантної
- `unclear` — недостатньо інформації або складно визначити

**Важливо:** повідомлення про трагічну подію може бути `neutral`,
якщо виклад суто фактологічний. Реакції аудиторії не визначають мітку.

## 2. manual_intensity — 0, 1 або 2
- `0` — немає вираженого емоційного оформлення
- `1` — помірно виражене
- `2` — сильне (інтенсивна лексика, експресивне оформлення)
Якщо `manual_emotion=neutral`, зазвичай `manual_intensity=0`.

## 3. manual_rhetorical_cues — можна декілька через `;`
- `none` — немає явно виражених прийомів
- `urgency` — наголошення на терміновості («негайно», «зараз»), що виходить за суто часову інформацію
- `fear_appeal` — спроба переконати через наголошення на загрозі
- `emotive_labeling` — емоційно заряджені характеристики чи оціночні ярлики
- `call_to_action` — явний заклик щось зробити (напр. пожертвувати кошти)
- `unclear` — неможливо визначити
Не називайте ці прийоми автоматично маніпуляцією.

## 4. manual_topic — одна категорія
`war_security`, `politics`, `economy`, `technology`, `daily_life`,
`advertising_donations`, `other`, `unclear`.

## 5. manual_uncertain
`yes` — маєте сумніви щодо мітки, інакше `no`.

Можна почати з 20–30 рядків для узгодження правил, а потім дорозмітити решту.
Після розмітки збережіть файл як CSV UTF-8 (не змінюйте `annotation_id`).
Для наукової роботи бажано мати другу незалежну розмітку частини записів,
щоб оцінити узгодженість, а не покладатися тільки на одну інтуїтивну оцінку.

**Методологічні обмеження:**
- 2026-09 означає лише UTC-день; порядок повідомлень в один день не відомий.
- Немає справжнього графа пересилань або підписників.
- Emoji-реакції агреговані без часу їх надходження та без кількості переглядів.
- Мітки контенту описують форму повідомлення; вони не доводять психологічний ефект.
- Наявність одного каналу в декількох темах не дає автоматичних причинних зв'язків.
'''


def run(source: Path, output: Path, sample_size: int, seed: int, min_len: int, max_len: int):
    if not source.is_file():
        raise SystemExit(f'Файл не знайдено: {source}\n'
                         'Запусти етап 2, щоб створити results_september_stage2/posts_prepared.jsonl, '
                         'або передай --file ШЛЯХ_ДО_ФАЙЛУ.')
    output.mkdir(parents=True, exist_ok=True)
    rows, stats = load_prepared(source, min_len, max_len)
    selected = stratified_sample(rows, sample_size, seed)
    rng = random.Random(seed + 4567)
    # Holdout assignments in key only. Human labeling stays blind to split and reactions.
    holdout_ids = set(r['post_row'] for r in rng.sample(selected, round(0.25 * len(selected))))

    blind_cols = ['annotation_id', 'text_for_label', 'manual_emotion', 'manual_intensity',
                  'manual_rhetorical_cues', 'manual_topic', 'manual_uncertain', 'manual_notes']
    key_cols = ['annotation_id', 'post_row', 'channel', 'date_utc', 'week', 'split',
                'reaction_count', 'comments_count', 'emoji_anger_share',
                'emoji_sadness_share', 'emoji_fear_share', 'emoji_laughter_share',
                'emoji_other_share']
    with (output / 'annotation_blind.csv').open('w', encoding='utf-8-sig', newline='') as fa, \
         (output / 'annotation_key_DO_NOT_OPEN_BEFORE_LABELING.csv').open('w', encoding='utf-8-sig', newline='') as fk:
        wa = csv.DictWriter(fa, fieldnames=blind_cols)
        wk = csv.DictWriter(fk, fieldnames=key_cols)
        wa.writeheader()
        wk.writeheader()
        for index, rec in enumerate(selected, 1):
            annotation_id = f'A{index:04d}'
            wa.writerow({'annotation_id': annotation_id, 'text_for_label': rec['text']})
            wk.writerow({'annotation_id': annotation_id,
                         **{col: rec.get(col, '') for col in key_cols[1:] if col != 'split'},
                         'split': 'test_holdout' if rec['post_row'] in holdout_ids else 'dev'})

    aggregate_rows = channel_day_markers(rows, output / 'channel_day_markers.csv')
    summary = {
        'stage': 3,
        'source': str(source),
        'method': 'reactions-blind manual annotation; channel-balanced and week-diverse pilot',
        'parameters': {'sample_size_requested': sample_size, 'seed': seed,
                       'annotation_min_chars': min_len, 'annotation_max_chars': max_len},
        'counts': dict(stats),
        'annotation_sample_size': len(selected),
        'annotation_unique_channels': len(set(r['channel'] for r in selected)),
        'annotation_week_distribution': dict(sorted(Counter(r['week'] for r in selected).items())),
        'holdout_sample_size': len(holdout_ids),
        'channel_day_marker_rows': aggregate_rows,
        'note': 'Manual emotions are still blank. Surface markers are NOT psychological effects. '
                'Any relation between reactions and content must be assessed only after blind labeling; '
                'no reaction time or views are available.',
    }
    (output / 'stage3_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'annotation_guide.md').write_text(GUIDE, encoding='utf-8')
    print(f'Прочитано: {stats["all_prepared_rows"]:,} підготовлених записів.')
    print(f'Придатних для пілотної розмітки: {len(rows):,}.')
    print(f'Вибрано: {len(selected)} текстів з {summary["annotation_unique_channels"]} назв каналів.')
    print(f'Файли збережені в: {output.resolve()}')
    print('Надішли: stage3_summary.json і ПЕРШІ 20–30 РОЗМІЧЕНИХ рядків з annotation_blind.csv.')
    print('Не надсилай повний posts_prepared.jsonl і поки не відкривай annotation_key_DO_NOT_OPEN_BEFORE_LABELING.csv.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--min-len', type=int, default=40)
    parser.add_argument('--max-len', type=int, default=1400)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error('--sample-size must be >= 1')
    if not (1 <= args.min_len < args.max_len):
        parser.error('Expected 1 <= --min-len < --max-len')
    run(args.file, args.out, args.sample_size, args.seed, args.min_len, args.max_len)


if __name__ == '__main__':
    main()
