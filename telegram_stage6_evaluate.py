#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЕТАП 6 ІНДЗ: оцінка текстового класифікатора на погоджених 90 нових текстах.

Запуск із папки PythonProject3:
    python telegram_stage6_evaluate.py

Без додаткових бібліотек (стандартний Python 3.9+).

ВАЖЛИВО: це оцінка на AI-assisted мітках, переглянутих/погоджених користувачем,
а НЕ незалежна експертна анотація. Контрольна вибірка навмисно збагачена за
скором моделі та збалансована за назвами каналів; не можна переносити її
метрики або частку класів на всі публікації Telegram.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
DEFAULT_LABELS = BASE / "telegram_stage5_annotation_90_filled.csv"
DEFAULT_KEY = BASE / "results_september_stage5" / "stage5_audit_key_DO_NOT_OPEN_BEFORE_LABELING.csv"
DEFAULT_PRED = BASE / "results_september_stage5" / "stage5_pilot_predictions.csv"
DEFAULT_OUT = BASE / "results_september_stage6"

EMOTIONS = {"neutral", "anger", "sadness", "fear", "hope_joy", "mixed", "unclear"}
TOPICS = {"war_security", "politics", "economy", "technology", "daily_life", "advertising_donations", "other", "unclear"}
CUES = {"none", "urgency", "fear_appeal", "emotive_labeling", "call_to_action", "unclear"}
STRATA = ("lower_third", "middle_third", "upper_third")


def die(message: str):
    raise SystemExit("ПОМИЛКА: " + message)


def csv_rows(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        die(f"Не знайдено {label}: {path}\n"
            "Перевір папку PythonProject3 або назву файлу (не повторюй Stage 5).")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            die(f"{label}: відсутні стовпці {missing}")
        rows = list(reader)
    if not rows:
        die(f"{label} порожній")
    return rows


def csv_write(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def id_map(rows: list[dict], name: str) -> dict[str, dict]:
    mapped = {}
    for r in rows:
        aid = (r.get("annotation_id") or "").strip()
        if not aid or aid in mapped:
            die(f"У файлі {name} є порожні чи повторні annotation_id: {aid!r}")
        mapped[aid] = r
    return mapped


def validate_annotations(rows: list[dict]) -> dict[str, dict]:
    mapped = id_map(rows, "розмітки")
    for aid, r in mapped.items():
        emo = (r["manual_emotion"] or "").strip().lower()
        if emo not in EMOTIONS - {"unclear"}:
            die(f"{aid}: manual_emotion має бути однією із категорій, але отримано {emo!r}")
        intensity = (r["manual_intensity"] or "").strip()
        if intensity not in {"0", "1", "2"}:
            die(f"{aid}: manual_intensity має бути 0, 1 або 2")
        topic = (r["manual_topic"] or "").strip()
        if topic not in TOPICS:
            die(f"{aid}: невідома тема {topic!r}")
        uncertain = (r["manual_uncertain"] or "").strip().lower()
        if uncertain not in {"yes", "no"}:
            die(f"{aid}: manual_uncertain має бути yes/no")
        cues = (r["manual_rhetorical_cues"] or "").strip()
        parts = [p.strip() for p in cues.split(";")]
        if len(parts) != len(set(parts)) or any(p not in CUES for p in parts):
            die(f"{aid}: неправильно задані риторичні прийоми: {cues!r}")
        if len(parts) > 1 and ("none" in parts or "unclear" in parts):
            die(f"{aid}: none/unclear не можна поєднувати з іншими прийомами")
        if not (r["text_for_label"] or "").strip():
            die(f"{aid}: порожній текст")
    return mapped


def get_keys(rows: list[dict]) -> dict[str, dict]:
    mapped = id_map(rows, "Stage 5 key")
    seen_posts = set()
    for aid, r in mapped.items():
        if (r["score_stratum"] or "").strip() not in STRATA:
            die(f"{aid}: невідома страта score_stratum")
        if (r["predicted_binary"] or "").strip() not in {"neutral", "emotional"}:
            die(f"{aid}: невідомий прогноз predicted_binary")
        try:
            score = float(r["model_score_UNCALIBRATED"])
            post = int(r["post_row"])
        except (TypeError, ValueError):
            die(f"{aid}: некоректний score або post_row")
        if not 0 <= score <= 1 or post in seen_posts:
            die(f"{aid}: score не у [0,1] або повторюється post_row")
        seen_posts.add(post)
        expected = "emotional" if score >= .5 else "neutral"
        if expected != r["predicted_binary"].strip():
            die(f"{aid}: прогноз не відповідає порогу 0.5 — перевір Stage 5 key")
    return mapped


def verify_predictions(path: Path, annotations: dict[str, dict], keys: dict[str, dict]):
    """Independent consistency check: Stage5 key points to exactly the scored text.

    Uses streaming, no need to read >25k texts into memory.
    """
    if not path.is_file():
        die(f"Не знайдено stage5_pilot_predictions.csv: {path}")
    selected = {int(r["post_row"]): aid for aid, r in keys.items()}
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        need = {"post_row", "text_for_model", "predicted_binary_PILOT", "model_score_UNCALIBRATED"}
        if not need.issubset(fields):
            die(f"Stage 5 predictions не має стовпців {sorted(need - fields)}")
        for r in reader:
            try:
                post = int(r["post_row"])
            except (ValueError, TypeError):
                die("Stage 5 predictions: неправильний post_row")
            if post not in selected:
                continue
            aid = selected[post]
            if post in seen:
                die(f"Stage5 predictions: post_row {post} повторюється")
            seen.add(post)
            reference_text = annotations[aid]["text_for_label"].strip()
            if reference_text != r["text_for_model"].strip():
                die(f"{aid}: текст у погодженій розмітці відрізняється від "
                    "тексту в Stage 5. Потрібно узгодити джерела, не оцінювати інший текст.")
            if r["predicted_binary_PILOT"].strip() != keys[aid]["predicted_binary"].strip():
                die(f"{aid}: прогноз у key і predictions різниться")
            if abs(float(r["model_score_UNCALIBRATED"]) - float(keys[aid]["model_score_UNCALIBRATED"])) > 0.00002:
                die(f"{aid}: різний score у Stage 5 key і predictions")
    if seen != set(selected):
        die(f"У stage5_pilot_predictions.csv не знайдені {len(set(selected) - seen)} обрані рядки")
    return len(seen)


def binary_metrics(rows: list[dict], *, always_neutral=False) -> dict:
    tn = fp = fn = tp = 0
    for r in rows:
        truth = r["reference_binary"] == "emotional"
        predicted = (not always_neutral) and r["predicted_binary"] == "emotional"
        if truth and predicted:
            tp += 1
        elif truth and not predicted:
            fn += 1
        elif not truth and predicted:
            fp += 1
        else:
            tn += 1
    n = len(rows)
    safe = lambda num, denom: round(num / denom, 4) if denom else None
    prec_pos = safe(tp, tp + fp) if tp + fp else 0.0
    rec_pos = safe(tp, tp + fn)
    prec_neg = safe(tn, tn + fn) if tn + fn else 0.0
    rec_neg = safe(tn, tn + fp)
    # use unrounded for exact final F1
    f1_pos = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else (None if not (tp + fn) else 0.0)
    f1_neg = (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else (None if not (tn + fp) else 0.0)
    result = {
        "samples": n,
        "neutral_support": tn + fp,
        "emotional_support": tp + fn,
        "predicted_emotional": tp + fp,
        "accuracy": safe(tn + tp, n),
        "balanced_accuracy": round((rec_pos + rec_neg) / 2, 4) if rec_pos is not None and rec_neg is not None else None,
        "emotional_precision": prec_pos,
        "emotional_recall": rec_pos,
        "emotional_f1": round(f1_pos, 4) if f1_pos is not None else None,
        "neutral_f1": round(f1_neg, 4) if f1_neg is not None else None,
        "macro_f1": round((f1_pos + f1_neg) / 2, 4) if f1_pos is not None and f1_neg is not None else None,
        "confusion_matrix_true_rows_pred_columns_neutral_emotional": [[tn, fp], [fn, tp]],
    }
    return result


def auc_rank(rows: list[dict]):
    """ROC AUC from pairwise model-score ranks (not probability calibration)."""
    positive = [r["score"] for r in rows if r["reference_binary"] == "emotional"]
    negative = [r["score"] for r in rows if r["reference_binary"] == "neutral"]
    if not positive or not negative:
        return None
    good = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return round(good / (len(positive) * len(negative)), 4)


def summarize_by(rows: list[dict], field: str, names: list[str]) -> list[dict]:
    return [{field: name, **binary_metrics([r for r in rows if r[field] == name])}
            for name in names if any(r[field] == name for r in rows)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--key", type=Path, default=DEFAULT_KEY)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    labels = validate_annotations(csv_rows(
        args.labels,
        {"annotation_id", "text_for_label", "manual_emotion", "manual_intensity", "manual_rhetorical_cues", "manual_topic", "manual_uncertain", "manual_notes"},
        "погоджену розмітку"))
    keys = get_keys(csv_rows(
        args.key,
        {"annotation_id", "post_row", "score_stratum", "predicted_binary", "model_score_UNCALIBRATED", "channel", "date_utc"},
        "Stage 5 key"))
    if set(labels) != set(keys):
        die("ID погодженої розмітки і Stage5 key не збігаються. "
            f"Лише в розмітці: {sorted(set(labels) - set(keys))[:5]}; "
            f"лише в key: {sorted(set(keys) - set(labels))[:5]}. "
            "Не запускай Stage 5 ще раз: використовуй оригінальний key.")
    verified = verify_predictions(args.predictions, labels, keys)

    merged = []
    for aid, ann in labels.items():
        key = keys[aid]
        truth = "neutral" if ann["manual_emotion"].strip() == "neutral" else "emotional"
        pred = key["predicted_binary"].strip()
        merged.append({
            "annotation_id": aid,
            "post_row": key["post_row"],
            "score_stratum": key["score_stratum"].strip(),
            "model_score_UNCALIBRATED": key["model_score_UNCALIBRATED"],
            "score": float(key["model_score_UNCALIBRATED"]),
            "reference_binary": truth,
            "predicted_binary": pred,
            "correct": "yes" if truth == pred else "no",
            "manual_emotion": ann["manual_emotion"].strip(),
            "manual_intensity": ann["manual_intensity"].strip(),
            "manual_rhetorical_cues": ann["manual_rhetorical_cues"].strip(),
            "manual_topic": ann["manual_topic"].strip(),
            "manual_uncertain": ann["manual_uncertain"].strip().lower(),
            "manual_notes": ann["manual_notes"].strip(),
            "channel": key["channel"],
            "date_utc": key["date_utc"],
            "text_for_label": ann["text_for_label"],
        })
    merged.sort(key=lambda x: x["annotation_id"])

    accuracy = binary_metrics(merged)
    majority = binary_metrics(merged, always_neutral=True)
    auc = auc_rank(merged)
    groups_strata = summarize_by(merged, "score_stratum", list(STRATA))
    groups_uncertainty = summarize_by(merged, "manual_uncertain", ["no", "yes"])
    false_positives = [r for r in merged if r["reference_binary"] == "neutral" and r["predicted_binary"] == "emotional"]
    false_negatives = [r for r in merged if r["reference_binary"] == "emotional" and r["predicted_binary"] == "neutral"]
    errors = [r for r in merged if r["correct"] == "no"]
    by_emotion = [{"emotion": emo, "count": ct,
                   "correct": sum(r["correct"] == "yes" for r in merged if r["manual_emotion"] == emo),
                   "missed_emotional": sum(r["predicted_binary"] == "neutral" for r in merged if r["manual_emotion"] == emo) if emo != "neutral" else ""}
                  for emo, ct in sorted(Counter(r["manual_emotion"] for r in merged).items())]

    args.out.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in merged[0] if c != "score"]
    csv_write(args.out / "stage6_all_90_comparison.csv", export_cols, merged)
    csv_write(args.out / "stage6_errors_for_review.csv", export_cols, errors)
    group_cols = ["score_stratum", "samples", "neutral_support", "emotional_support",
                  "predicted_emotional", "accuracy", "balanced_accuracy", "emotional_precision",
                  "emotional_recall", "emotional_f1", "neutral_f1", "macro_f1"]
    csv_write(args.out / "stage6_by_score_stratum.csv", group_cols, groups_strata)
    csv_write(args.out / "stage6_by_label_uncertainty.csv", ["manual_uncertain", *group_cols[1:]], groups_uncertainty)
    csv_write(args.out / "stage6_by_original_emotion.csv", list(by_emotion[0]), by_emotion)
    tn, fp = accuracy["confusion_matrix_true_rows_pred_columns_neutral_emotional"][0]
    fn, tp = accuracy["confusion_matrix_true_rows_pred_columns_neutral_emotional"][1]
    csv_write(args.out / "stage6_confusion_matrix.csv",
              ["reference_label", "predicted_neutral", "predicted_emotional"],
              [{"reference_label": "neutral", "predicted_neutral": tn, "predicted_emotional": fp},
               {"reference_label": "emotional", "predicted_neutral": fn, "predicted_emotional": tp}])

    summary = {
        "stage": 6,
        "objective": "EXPLORATORY external-to-training audit of binary textual affect framing",
        "labels_source": str(args.labels),
        "audit_key_source": str(args.key),
        "predictions_crosscheck_source": str(args.predictions),
        "audit_inputs_cross_checked": verified,
        "annotation_provenance": "AI-assisted labels, subsequently reviewed and accepted by user; not independent expert / double-blind ground truth",
        "sampling_design": "90 model-score-enriched, channel-balanced audit items drawn from 3 quantile score thirds; not prevalence representative",
        "reference_emotion_counts": dict(sorted(Counter(r["manual_emotion"] for r in merged).items())),
        "reference_binary_counts": dict(sorted(Counter(r["reference_binary"] for r in merged).items())),
        "manual_uncertain_counts": dict(sorted(Counter(r["manual_uncertain"] for r in merged).items())),
        "model_metrics_on_enriched_audit": accuracy,
        "baseline_always_neutral_on_same_audit": majority,
        "rank_auc_roc_on_enriched_audit": auc,
        "error_analysis": {
            "n_errors": len(errors),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "error_annotation_ids": [r["annotation_id"] for r in errors],
            "errors_marked_uncertain": sum(r["manual_uncertain"] == "yes" for r in errors),
        },
        "by_score_stratum": groups_strata,
        "by_label_uncertainty": groups_uncertainty,
        "by_original_emotion": by_emotion,
        "limitations": [
            "Labels are AI-generated drafts accepted by the user, NOT independent human expert ground truth.",
            "Audit was selected by existing model score and channel balancing; performance estimates do NOT transfer directly to all Telegram posts.",
            "Model scores are not calibrated probabilities; ROC AUC is ranking-only, not probability calibration.",
            "The audit is new text records, but semantically near-identical news from different channels may still overlap with training.",
            "For robust external validation, use independent annotators and a time-forward or event-grouped test with appropriate sampling.",
            "Do not tune thresholds or models on this audit and then report the same audit as an untouched holdout.",
            "Results describe TEXTUAL framing; they do not measure human psychological effects, contagion, real repost diffusion or causal impact.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "stage6_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fmt = lambda x: "н/д" if x is None else f"{100*x:.1f}%"
    report = f"""# Етап 6. Порівняння з погодженою розміткою 90 повідомлень

**Статус:** пілотна перевірка (НЕ незалежний експертний тест і НЕ вимір психологічного впливу).

Мітки: ШІ підготував оцінки текстів, користувач їх переглянув і погодив.
Зразок: {len(merged)} нових повідомлень, збалансованих за каналами та третинами скору старої моделі.

## Основні показники

| Показник | NLP-модель | Завжди `neutral` |
|---|---:|---:|
| Accuracy | {fmt(accuracy['accuracy'])} | {fmt(majority['accuracy'])} |
| Balanced accuracy | {fmt(accuracy['balanced_accuracy'])} | {fmt(majority['balanced_accuracy'])} |
| Precision для emotional | {fmt(accuracy['emotional_precision'])} | {fmt(majority['emotional_precision'])} |
| Recall для emotional | {fmt(accuracy['emotional_recall'])} | {fmt(majority['emotional_recall'])} |
| F1 для emotional | {fmt(accuracy['emotional_f1'])} | {fmt(majority['emotional_f1'])} |
| Macro-F1 | {fmt(accuracy['macro_f1'])} | {fmt(majority['macro_f1'])} |

Матриця помилок: TN={tn}, FP={fp}, FN={fn}, TP={tp}.
Загалом помилок: **{len(errors)}**; із них {len(false_positives)} хибних спрацьовувань
і {len(false_negatives)} пропущених емоційних текстів.
У сумнівних щодо мітки прикладах: {sum(r['manual_uncertain']=='yes' for r in merged)} із {len(merged)}.

## Науково коректне трактування

- Це оцінка **мовного оформлення**, а не емоцій аудиторії чи фактичного інформаційно-психологічного впливу.
- Вибірку спеціально підбирали за прогнозом моделі, тому вона **не є випадковою репрезентативною** вибіркою вересневого корпусу.
- Тому навіть висока точність **не доводить** таку саму точність на всіх 25 852 текстах.
- Оцінки користувача — погодження AI-assisted міток, а не незалежне подвійне анотування.
- Метрики Stage 4 і Stage 6 отримано на вибірках із різними механізмами відбору; не варто трактувати різницю як поліпшення / погіршення моделі.
- Прогнози моделі не слід використовувати як фактичні психометричні показники.

## Наступний крок

Для ATSM-IM створити **проксі-граф тематичної близькості** між каналами за схожими
текстами та датами, із часовими ознаками на рівні доби. Потім на цьому
**змодельованому (не спостережуваному) графі** порівняти адаптивну та прості
стратегії вибору вузлів у Монте-Карло-симуляціях із однаковими бюджетами.
Окремо зафіксувати експериментальні припущення, baselines, абляції та seeds.
"""
    (args.out / "stage6_report_UA.md").write_text(report, encoding="utf-8")
    print("\nЕтап 6 виконано: перевірено", verified, "повідомлень")
    print("Accuracy:", fmt(accuracy["accuracy"]), "| Balanced accuracy:", fmt(accuracy["balanced_accuracy"]))
    print("Emotional F1:", fmt(accuracy["emotional_f1"]), "| Помилок:", len(errors))
    print("TN, FP, FN, TP:", tn, fp, fn, tp)
    print("Папка з результатами:", args.out.resolve())
    print("Для наступного кроку надішли stage6_summary.json і stage6_errors_for_review.csv")
    print("УВАГА: це оцінка на погоджених AI-assisted мітках, не незалежна експертна перевірка.")


if __name__ == "__main__":
    main()
