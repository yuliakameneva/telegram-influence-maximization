#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4: text-only, binary emotion-expression classifier (September 2026 Telegram).

Train/dev and test_holdout assignments are read from Stage 3's key file, but
NONE of the reactions, comments, channel names, dates or emoji shares are used
as ML features. Model input = text_for_label only.

Install: python -m pip install scikit-learn matplotlib joblib
Run:     python telegram_stage4_train.py
Output:  results_september_stage4/

This is a small pilot on 100 AI-assisted, user-reviewed labels. It does not
establish psychological causation, nor does it justify production inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

EMOTIONS = {"neutral", "anger", "fear", "sadness", "hope_joy", "mixed", "unclear"}
DEFAULT_LABELS = Path("telegram_labels_reviewed_100.csv")
DEFAULT_KEY = Path("results_september_stage3/annotation_key_DO_NOT_OPEN_BEFORE_LABELING.csv")
DEFAULT_OUT = Path("results_september_stage4")


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(
            f"Не знайдено файл: {path}\n"
            f"Робоча папка: {Path.cwd()}\n"
            "Перевір назву файла та шлях. Файл із мітками поклади поруч зі скриптом."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_records(label_file: Path, split_file: Path) -> list[dict]:
    labels = load_csv(label_file)
    keys = load_csv(split_file)
    if not labels or not keys:
        raise SystemExit("Порожній CSV із мітками або розподілом dev/test.")
    mandatory_labels = {"annotation_id", "text_for_label", "manual_emotion", "manual_uncertain"}
    mandatory_key = {"annotation_id", "split"}
    if not mandatory_labels.issubset(labels[0]) or not mandatory_key.issubset(keys[0]):
        raise SystemExit("Невірні заголовки. Перевір, чи використовуєш CSV із етапів 3 і 4.")
    ids = [r["annotation_id"].strip() for r in labels]
    key_ids = [r["annotation_id"].strip() for r in keys]
    if len(ids) != len(set(ids)) or len(key_ids) != len(set(key_ids)):
        raise SystemExit("Знайдено повторювані annotation_id у CSV-файлах.")
    key_by_id = {r["annotation_id"].strip(): r for r in keys}
    if set(ids) != set(key_by_id):
        missing_key = set(ids) - set(key_by_id)
        missing_label = set(key_by_id) - set(ids)
        raise SystemExit(
            "Не збігаються ID розмітки й Stage 3 key. "
            f"Відсутні в key: {sorted(missing_key)[:4]}; "
            f"відсутні у labels: {sorted(missing_label)[:4]}. "
            "Не запускай stage3 повторно з іншими seed/sample-size."
        )
    result = []
    for r in labels:
        annotation_id = r["annotation_id"].strip()
        split = key_by_id[annotation_id]["split"].strip()
        emotion = r["manual_emotion"].strip().lower()
        unsure = r["manual_uncertain"].strip().lower()
        text = r["text_for_label"].strip()
        if split not in {"dev", "test_holdout"}:
            raise SystemExit(f"{annotation_id}: некоректний split '{split}'.")
        if emotion not in EMOTIONS:
            raise SystemExit(f"{annotation_id}: некоректна емоція '{emotion}'.")
        if emotion == "unclear":
            raise SystemExit(f"{annotation_id}: для невизначеної емоції поки немає надійного target; "
                             "виправ оцінку або розглянь її виключення окремо.")
        if unsure not in {"yes", "no"} or not text:
            raise SystemExit(f"{annotation_id}: відсутній текст або некоректне manual_uncertain.")
        result.append({
            "annotation_id": annotation_id,
            "text": text,
            "emotion": emotion,
            "label": 0 if emotion == "neutral" else 1,
            "split": split,
            "uncertain": unsure,
            "annotation_origin": r.get("annotation_origin", ""),
            "reviewed_by_user": r.get("reviewed_by_user", ""),
        })
    return result


def build_models():
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    def features():
        return FeatureUnion([
            ("word", TfidfVectorizer(
                lowercase=True, analyzer="word", ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b", min_df=2,
                max_features=12000, sublinear_tf=True)),
            ("char", TfidfVectorizer(
                lowercase=True, analyzer="char", ngram_range=(3, 5),
                min_df=2, max_features=20000, sublinear_tf=True)),
        ])

    return {
        "dummy_majority": Pipeline([
            ("vect", TfidfVectorizer(analyzer="char", ngram_range=(2, 3))),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "tfidf_logreg": Pipeline([
            ("features", features()),
            ("clf", LogisticRegression(
                class_weight="balanced", solver="liblinear", max_iter=2000,
                random_state=2026)),
        ]),
        "tfidf_linear_svm": Pipeline([
            ("features", features()),
            ("clf", LinearSVC(class_weight="balanced", C=1.0, random_state=2026)),
        ]),
    }


def measure(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score,
    )
    # Label 1 = explicitly emotional; label 0 = neutral.
    mat = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "samples": len(y_true),
        "neutral_support": int(sum(y == 0 for y in y_true)),
        "emotional_support": int(sum(y == 1 for y in y_true)),
        "predicted_emotional": int(sum(y == 1 for y in y_pred)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": (round(float(balanced_accuracy_score(y_true, y_pred)), 4)
                              if len(set(y_true)) == 2 else None),
        "emotion_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "emotion_recall": (round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
                           if 1 in y_true else None),
        "emotion_f1": (round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
                       if 1 in y_true else None),
        "macro_f1": (round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
                     if len(set(y_true)) == 2 else None),
        "confusion_matrix_true_rows_pred_columns_neutral_emotional": mat,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def draw_plots(out: Path, train: list[dict], test: list[dict], mat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = ["Train/dev", "Test holdout"]
    neutrals = [sum(r["label"] == 0 for r in chunk) for chunk in (train, test)]
    emotional = [sum(r["label"] == 1 for r in chunk) for chunk in (train, test)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, neutrals, label="Neutral")
    ax.bar(x, emotional, bottom=neutrals, label="Emotional")
    ax.set_ylabel("Number of annotated posts")
    ax.set_title("Pilot label distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stage4_label_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([0, 1], ["Neutral", "Emotional"])
    ax.set_yticks([0, 1], ["Neutral", "Emotional"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Reviewed label")
    ax.set_title("Held-out confusion matrix")
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center",
                    color="white" if mat[i][j] > max(max(row) for row in mat) / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "stage4_holdout_confusion.png", dpi=150)
    plt.close(fig)


def run(labels_path: Path, key_path: Path, out: Path, seed: int, folds: int):
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import joblib

    records = load_records(labels_path, key_path)
    dev = [r for r in records if r["split"] == "dev"]
    test = [r for r in records if r["split"] == "test_holdout"]
    if len(dev) < 20 or len(test) < 10:
        raise SystemExit(f"Замало даних: dev={len(dev)} / holdout={len(test)}.")
    dev_labels = Counter(r["label"] for r in dev)
    if min(dev_labels.values(), default=0) < 3 or len(dev_labels) != 2:
        raise SystemExit(f"Замало прикладів одного з двох класів у dev: {dict(dev_labels)}.")
    n_splits = min(folds, min(dev_labels.values()))
    if n_splits < 3:
        raise SystemExit("Мінімум три складки CV; додай більше погоджених прикладів.")

    out.mkdir(parents=True, exist_ok=True)
    x_dev = [r["text"] for r in dev]
    y_dev = [r["label"] for r in dev]
    x_test = [r["text"] for r in test]
    y_test = [r["label"] for r in test]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = build_models()

    comparisons = []
    for name, model in models.items():
        preds = cross_val_predict(model, x_dev, y_dev, cv=cv, method="predict", n_jobs=1)
        scores = measure(y_dev, preds)
        comparisons.append({"model": name, **scores})
        print(f"CV {name}: emotional F1={scores['emotion_f1']:.3f}; "
              f"balanced accuracy={scores['balanced_accuracy']:.3f}; "
              f"accuracy={scores['accuracy']:.3f}")

    # Selection based ONLY on development-set cross-validation;
    # holdout is neither used to choose model nor tune parameters.
    learner_rows = [r for r in comparisons if r["model"] != "dummy_majority"]
    chosen = max(learner_rows,
                 key=lambda r: (r["emotion_f1"], r["balanced_accuracy"], r["macro_f1"]))
    selected_name = chosen["model"]
    selected_model = clone(models[selected_name]).fit(x_dev, y_dev)
    pred_test = [int(x) for x in selected_model.predict(x_test)]
    test_score = measure(y_test, pred_test)
    print(f"HOLDOUT {selected_name}: emotional F1={test_score['emotion_f1']:.3f}; "
          f"balanced accuracy={test_score['balanced_accuracy']:.3f}; "
          f"accuracy={test_score['accuracy']:.3f}")

    # Same holdout also evaluates majority baseline for honest comparison.
    dummy = clone(models["dummy_majority"]).fit(x_dev, y_dev)
    baseline_test_score = measure(y_test, [int(x) for x in dummy.predict(x_test)])

    predictions = []
    errors = []
    for r, p in zip(test, pred_test):
        result = {
            "annotation_id": r["annotation_id"],
            "reference_binary": "emotional" if r["label"] else "neutral",
            "predicted_binary": "emotional" if p else "neutral",
            "correct": "yes" if r["label"] == p else "no",
            "original_emotion_label": r["emotion"],
            "manual_uncertain": r["uncertain"],
            "text": r["text"],
        }
        predictions.append(result)
        if r["label"] != p:
            errors.append(result)

    fieldnames = list(predictions[0])
    write_csv(out / "stage4_holdout_predictions.csv", fieldnames, predictions)
    write_csv(out / "stage4_errors_for_review.csv", fieldnames, errors)
    cv_fields = ["model", "samples", "neutral_support", "emotional_support",
                 "predicted_emotional", "accuracy", "balanced_accuracy", "emotion_precision",
                 "emotion_recall", "emotion_f1", "macro_f1"]
    write_csv(out / "stage4_cv_comparison.csv", cv_fields, comparisons)

    confident_idx = [i for i, r in enumerate(test) if r["uncertain"] == "no"]
    confident = (measure([y_test[i] for i in confident_idx], [pred_test[i] for i in confident_idx])
                 if confident_idx else None)
    # A tiny pilot is not evidence of reliable generalization or causal influence.
    summary = {
        "stage": 4,
        "task": "binary detection of explicit affective framing in text",
        "labels": {"0": "neutral", "1": "emotionally framed"},
        "total_rows": len(records),
        "reviewed_labels_source": str(labels_path),
        "split_source": str(key_path),
        "data_provenance": "AI-assisted draft, reviewed/accepted by user; not independently double-annotated",
        "train_dev_rows": len(dev),
        "test_holdout_rows": len(test),
        "train_dev_emotion_counts": dict(sorted(Counter(r["emotion"] for r in dev).items())),
        "test_holdout_emotion_counts": dict(sorted(Counter(r["emotion"] for r in test).items())),
        "total_uncertain_rows": sum(r["uncertain"] == "yes" for r in records),
        "test_holdout_uncertain_rows": sum(r["uncertain"] == "yes" for r in test),
        "test_holdout_confident_emotional_rows": sum(r["uncertain"] == "no" and r["label"] == 1 for r in test),
        "validation_warning": (
            "Holdout has no confident emotional examples; emotional recall on trusted-only "
            "holdout cannot be estimated. Add independently reviewed positive cases."
            if not any(r["uncertain"] == "no" and r["label"] == 1 for r in test)
            else "Holdout is small; treat scores as preliminary."),
        "feature_policy": "text_for_label ONLY (no emoji reaction counts, no comments, no channel, no timestamp)",
        "cross_validation": {"method": "StratifiedKFold", "folds": n_splits, "random_state": seed},
        "development_cv_comparison": comparisons,
        "model_selection": "max emotional F1, then balanced accuracy, then macro-F1; on dev CV only",
        "selected_model": selected_name,
        "selected_model_cv_scores": chosen,
        "selected_model_test_holdout_scores": test_score,
        "majority_baseline_test_holdout_scores": baseline_test_score,
        "test_confident_subset_scores_exploratory": confident,
        "limitations": [
            "100 samples with only 25 held out: results have high statistical uncertainty.",
            "AI-assisted, user-reviewed labels are not independent expert ground truth.",
            "Validation is text-based and binary; not yet multi-class emotion identification.",
            "The September pilot split is random, not a chronological future-event test.",
            "Similar news events can be present in both dev and test, making generalization optimistic.",
            "No reactions, comments, audiences or psychological change were predicted here.",
            "Do not extrapolate performance to all 27,815 posts without additional validation.",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "stage4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"pipeline": selected_model, "label_map": {0: "neutral", 1: "emotional"},
                 "purpose": "preliminary text-only binary classifier"},
                out / "stage4_binary_pilot_model.joblib")
    draw_plots(out, dev, test, test_score["confusion_matrix_true_rows_pred_columns_neutral_emotional"])
    method = f"""# Етап 4: звіт для ІНДЗ (пілот)

**Завдання:** бінарна класифікація емоційного оформлення повідомлень Telegram.

**Дані:** 100 перевірених користувачкою, ШІ-асистовано розмічених публікацій.
Dev/train: {len(dev)}; test_holdout: {len(test)}; із них невпевнених позначок: {summary['total_uncertain_rows']}.
**Ознаки:** лише текст повідомлення. Emoji-реакції, канали й дати не використовувалися для класифікації.

**Репрезентація:** TF-IDF ознаки слів (1–2-грам) та символьних фрагментів (3–5-грам).
**Моделі:** DummyClassifier (контроль), LogisticRegression (зважені класи), LinearSVC (зважені класи).
**Вибір моделі:** {n_splits}-fold stratified CV на dev, вибір за F1 емоційного класу.
**Незалежна перевірка:** відкладені {len(test)} текстів, **одне оцінювання після вибору**.

**Обрана модель:** `{selected_name}`.
**Результати на відкладеній вибірці:** emotional F1 = {test_score['emotion_f1']:.3f};
balanced accuracy = {test_score['balanced_accuracy']:.3f}; accuracy = {test_score['accuracy']:.3f}.
Для контрольного алгоритму (прогноз переважного класу): balanced accuracy = {baseline_test_score['balanced_accuracy']:.3f}.

**Надійність перевірки:** у відкладеній вибірці є тільки {summary['test_holdout_confident_emotional_rows']} впевнено позначених емоційних повідомлень.
Отже, якість розпізнавання впевнено розміченого емоційного класу окремо **не оцінена**.

**Обмеження:** вибірка мала; позначки ШІ-асистовані (після перевірки людиною),
а не незалежно двічі анотовані. Емоційне оформлення тексту не є доказом зміни
психологічного стану читачів. Перевірка ще не дозволяє стверджувати,
що модель правильно класифікує весь вересневий корпус. Реакції тут не є цільовими мітками.

**Наступний крок:** аналіз помилок, додаткові незалежно розмічені повідомлення,
визначення придатності багатомовної моделі; лише після цього — застосування
до повного корпусу та графових компонентів ATSM-IM.
"""
    (out / "stage4_methodology.md").write_text(method, encoding="utf-8")
    print(f"Готово. Результати: {out.resolve()}")
    if summary["test_holdout_confident_emotional_rows"] == 0:
        print("УВАГА: серед впевнених розміток holdout немає емоційних випадків;")
        print("не можна надійно оцінити виявлення емоцій без додаткових незалежних міток.")
    print("Надішли stage4_summary.json і stage4_errors_for_review.csv (можна ще графіки).")
    print("Файл .joblib залишай у себе; не завантажуй моделі з неперевірених джерел.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 3 or args.folds > 10:
        parser.error("--folds має бути від 3 до 10")
    try:
        run(args.labels, args.key, args.out, args.seed, args.folds)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Помилка даних: {exc}") from exc


if __name__ == "__main__":
    main()
