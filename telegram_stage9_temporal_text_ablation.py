#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Етап 9 ІНДЗ: ретроспективні СИНТЕТИЧНІ абляції часових і текстових ознак.

Запуск із кореня PythonProject3: python telegram_stage9_temporal_text_ablation.py
Потрібні лише локальні CSV попередніх етапів і файл етапу 8.
ЖОДНИХ запитів до інтернету, публікацій, реальних пересилань чи втручань.

Це демонстраційний алгоритм адаптивного вибору вузлів, а НЕ оцінка реального
інформаційно-психологічного впливу. Граф схожості контенту НЕ соціальний граф.
Значення NLP-моделі НЕ калібровані; усі коефіцієнти передачі синтетичні.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time

try:
    from telegram_stage8_simulate_adaptive import (
        read_graph, percentile_90, live_world, reach, static_mc_plan, stats,
    )
except ImportError as exc:
    raise SystemExit('Поклади telegram_stage8_simulate_adaptive.py поруч із цим скриптом.') from exc

ROOT = Path(__file__).resolve().parent
DEFAULT_EDGES = ROOT / 'results_september_stage7' / 'stage7_proxy_edges.csv'
DEFAULT_NODES = ROOT / 'results_september_stage7' / 'stage7_channel_nodes.csv'
DEFAULT_SIGNAL = ROOT / 'results_september_stage5' / 'stage5_channel_days_PILOT.csv'
DEFAULT_OUT = ROOT / 'results_september_stage9'

CONTEXT_DAYS = ['2026-09-10', '2026-09-18', '2026-09-25', '2026-09-30']
# Коефіцієнти — ВИНЯТКОВО гіпотези для аналізу чутливості, НЕ оцінки з даних.
SCENARIOS = [
    ('combined_assumption', .60, .60),
    ('time_only_assumption', .60, 0.),
    ('text_only_assumption', 0., .60),
    ('graph_only_control', 0., 0.),
    ('inverse_text_control', .60, -.60),
]
POLICIES = [
    'adaptive_temporal_text',
    'adaptive_time_only',
    'adaptive_text_only',
    'adaptive_graph_only',
    'static_graph_greedy',
    'random',
]


def die(message):
    raise SystemExit('ПОМИЛКА: ' + message)


def read_csv(path, required):
    if not path.is_file():
        die(f'Немає файлу {path}. Перевір попередні етапи.')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(set(reader.fieldnames or [])):
            die(f'У {path.name} немає очікуваних колонок: {sorted(set(required)-set(reader.fieldnames or []))}')
        return list(reader)


def save_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def load_signals(path, valid_names):
    cols = ['channel','date_utc','n_posts','mean_model_score_UNCALIBRATED']
    rows = read_csv(path, cols)
    values = {}
    for row in rows:
        ch = row['channel'].strip()
        try:
            dt = date.fromisoformat(row['date_utc'])
            n = int(row['n_posts'])
            score = float(row['mean_model_score_UNCALIBRATED'])
        except ValueError as exc:
            die(f'Некоректні значення channel-day: {row} ({exc})')
        if ch not in valid_names or not (date(2026,9,1) <= dt <= date(2026,9,30)):
            die(f'Невідомий канал або дата в {path.name}: {ch}, {dt}')
        if n < 0 or not math.isfinite(score) or not 0 <= score <= 1:
            die(f'Некоректні n_posts або score для {ch}, {dt}')
        key = (ch, dt)
        if key in values:
            die(f'Повторний запис {ch}, {dt} у channel-day CSV.')
        values[key] = (n, score)
    if len(values) < 30:
        die('Замало channel-day записів; потрібен результат етапу 5 для вересня.')
    matched = len({k[0] for k in values})
    if matched < max(3, len(valid_names)//2):
        die('Менше половини вузлів графа мають текстово-часові записи. Перевір, що файли з одного корпусу.')
    return values, matched


def clamp(x):
    return max(-1.0, min(1.0, x))


def make_context(names, signals, iso_date):
    current = date.fromisoformat(iso_date)
    # Ознаки обчислюємо ТІЛЬКИ з попередніх 7 календарних днів.
    last3 = [current - timedelta(days=i) for i in (1,2,3)]
    earlier4 = [current - timedelta(days=i) for i in (4,5,6,7)]
    all_recent = [signals[(ch,d)] for ch in names for d in last3 if (ch,d) in signals]
    global_score = (sum(n*sc for n,sc in all_recent)/sum(n for n,_ in all_recent)
                    if sum(n for n,_ in all_recent) > 0 else .5)
    context = []
    for name in names:
        nr = sum(signals.get((name,d), (0,0))[0] for d in last3)
        np = sum(signals.get((name,d), (0,0))[0] for d in earlier4)
        # Порівняння швидкості публікацій в останні 3 і попередні 4 дні.
        recent_rate = nr / 3
        previous_rate = np / 4
        time_signal = clamp((recent_rate - previous_rate) / max(3.0, previous_rate))
        if nr:
            mean_score = (sum((signals[(name,d)][0]*signals[(name,d)][1])
                              for d in last3 if (name,d) in signals) / nr)
            # Умовний індекс відносно середнього попередніх 3 днів; НЕ
            # імовірність страху, гніву або психологічного впливу.
            text_signal = clamp((mean_score - global_score) / .15)
        else:
            mean_score = None
            text_signal = 0.0
        context.append({
            'snapshot_utc_day': iso_date,
            'channel': name, 'prior_3d_posts': nr,
            'previous_4d_posts': np,
            'prior_3d_model_score_UNCALIBRATED': round(mean_score,6) if mean_score is not None else '',
            'time_activity_signal': round(time_signal,6),
            'relative_text_signal': round(text_signal,6),
        })
    return context


def weighted_arcs(base_edges, time_values, text_values, time_coef, text_coef, base_p):
    # Неорієнтований граф контенту замінюємо ДВОМА напрямками суто в симуляції.
    arcs = []
    for a,b,overlap_factor in base_edges:
        base = base_p * overlap_factor
        for src,dst in ((a,b),(b,a)):
            factor = (1+time_coef*time_values[src])*(1+text_coef*text_values[src])
            arcs.append((src,dst,min(.30,max(.0001,base*factor))))
    return arcs


def random_live_world(n, arcs, rng):
    adjacency = [0]*n
    for a,b,p in arcs:
        if rng.random() < p:
            adjacency[a] |= 1<<b
    return adjacency


def pick_adaptive(n, arcs, active_mask):
    # Адаптивний жадібний крок: умовна очікувана кількість ПРЯМИХ
    # нових активацій + активований вибраний вузол.
    scores = [0.] * n
    for src,dst,p in arcs:
        if not (active_mask & (1<<src)) and not (active_mask & (1<<dst)):
            scores[src] += p
    candidates = [v for v in range(n) if not (active_mask & (1<<v))]
    return min(candidates,key=lambda v:(-scores[v],v)) if candidates else None


def execute_adaptive(n, world, arcs, budget):
    active, levels = 0, []
    for _ in range(budget):
        nxt = pick_adaptive(n,arcs,active)
        if nxt is not None:
            active |= reach(nxt,world,active)
        levels.append(active.bit_count())
    return levels


def execute_static(world, chosen, budget):
    active, levels = 0, []
    for nxt in chosen[:budget]:
        active |= reach(nxt,world,active)
        levels.append(active.bit_count())
    return levels


def make_plot(path, rows, budget):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'skipped_no_matplotlib'
    r = [x for x in rows if x['scenario']=='combined_assumption' and x['budget']==budget]
    names = [x['algorithm'] for x in r]
    vals = [x['mean_activated'] for x in r]
    fig, ax = plt.subplots(figsize=(10,4.8))
    ax.bar(range(len(names)), vals)
    ax.set_xticks(range(len(names)),names,rotation=25,ha='right',fontsize=8)
    ax.set_ylim(0,max(vals)*1.2)
    ax.set_ylabel('Mean activated synthetic nodes')
    ax.set_title('Stage 9: combined hypothetical signals, 4 retrospective contexts')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)
    return 'saved'


def main():
    parser=argparse.ArgumentParser(description='Етап 9: часово-текстові абляції на синтетичному проксі-графі')
    parser.add_argument('--edges',type=Path,default=DEFAULT_EDGES)
    parser.add_argument('--nodes',type=Path,default=DEFAULT_NODES)
    parser.add_argument('--signals',type=Path,default=DEFAULT_SIGNAL)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--repetitions',type=int,default=120)
    parser.add_argument('--budget',type=int,default=4)
    parser.add_argument('--planning-worlds',type=int,default=200)
    parser.add_argument('--base-probability',type=float,default=.035)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if not (1<=args.budget<=25 and args.repetitions>0 and args.planning_worlds>0 and .001<=args.base_probability<=.1):
        parser.error('Перевір бюджет, повторення, планувальні світи й базову імовірність.')
    n, raw_edges = read_graph(args.edges,args.nodes)
    if args.budget>n:
        parser.error('Бюджет перевищує кількість вузлів.')
    node_data=read_csv(args.nodes,['channel'])
    names=sorted(x['channel'].strip() for x in node_data)
    signals,signal_ch_count=load_signals(args.signals,set(names))
    q90=percentile_90([x[2] for x in raw_edges])
    base_edges=[(a,b,.5+.9*min(1.,overlap/q90)) for a,b,overlap,_,_ in raw_edges]
    empty=[0.]*n
    graph_arcs=weighted_arcs(base_edges,empty,empty,0.,0.,args.base_probability)
    # Static greedy plan — offline, isolated from the evaluation-world random draws.
    # stage8.static_mc_plan expects undirected (a,b,p) tuples.
    planner_edges=[(a,b,min(.3,args.base_probability*f)) for a,b,f in base_edges]
    static_plan=static_mc_plan(n,planner_edges,args.planning_worlds,
                               random.Random(args.seed+140_000),args.budget)
    args.output.mkdir(exist_ok=True,parents=True)
    all_features=[]
    aggregate_scores=defaultdict(list)
    per_context_scores=defaultdict(list)
    processing_sec=defaultdict(float)
    print(f'Етап 9: {n} назв каналів, {len(raw_edges)} ребер, {len(signals)} записів channel-day.',flush=True)
    for cidx,when in enumerate(CONTEXT_DAYS):
        feature_rows=make_context(names,signals,when)
        all_features.extend(feature_rows)
        tm=[r['time_activity_signal'] for r in feature_rows]
        tx=[r['relative_text_signal'] for r in feature_rows]
        no_context=graph_arcs
        temporal=weighted_arcs(base_edges,tm,tx,.6,0.,args.base_probability)
        textual=weighted_arcs(base_edges,tm,tx,0.,.6,args.base_probability)
        combined=weighted_arcs(base_edges,tm,tx,.6,.6,args.base_probability)
        assumed={'adaptive_temporal_text':combined,'adaptive_time_only':temporal,
                 'adaptive_text_only':textual,'adaptive_graph_only':no_context}
        for sidx,(scenario, tcoef, xcoef) in enumerate(SCENARIOS):
            evaluation=weighted_arcs(base_edges,tm,tx,tcoef,xcoef,args.base_probability)
            for rep in range(args.repetitions):
                # One and the same simulated world for all policies in this repeat.
                rep_seed=args.seed+10_000_000*cidx+100_000*sidx+rep
                world=random_live_world(n,evaluation,random.Random(rep_seed))
                rng=random.Random(args.seed+7_000_000+cidx*100_000+sidx*20_000+rep)
                seq=list(range(n)); rng.shuffle(seq)
                for policy in POLICIES:
                    t0=time.perf_counter()
                    if policy=='static_graph_greedy':
                        outcomes=execute_static(world,static_plan,args.budget)
                    elif policy=='random':
                        outcomes=execute_static(world,seq,args.budget)
                    else:
                        outcomes=execute_adaptive(n,world,assumed[policy],args.budget)
                    processing_sec[policy]+=time.perf_counter()-t0
                    for k,y in enumerate(outcomes,1):
                        aggregate_scores[(scenario,policy,k)].append(y)
                        per_context_scores[(when,scenario,policy,k)].append(y)
            print(f'[{cidx+1}/4] {when} — {scenario}: OK',flush=True)
    results=[]
    for scen,_,_ in SCENARIOS:
        for k in range(1,args.budget+1):
            ref=aggregate_scores[(scen,'adaptive_graph_only',k)]
            for policy in POLICIES:
                vals=aggregate_scores[(scen,policy,k)]
                mean,lo,hi=stats(vals)
                diff=[a-b for a,b in zip(vals,ref)]
                dm,dlo,dhi=stats(diff)
                results.append({'scenario':scen,'algorithm':policy,'budget':k,
                    'repetitions':len(vals),'mean_activated':mean,'ci95_lower':max(0.,lo),
                    'ci95_upper':min(n,hi),'gain_vs_adaptive_graph':dm,
                    'gain_ci95_lower':dlo,'gain_ci95_upper':dhi})
    detail=[]
    for when in CONTEXT_DAYS:
        for scen,_,_ in SCENARIOS:
            for k in range(1,args.budget+1):
                for policy in POLICIES:
                    vals=per_context_scores[(when,scen,policy,k)]
                    mean,lo,hi=stats(vals)
                    detail.append({'snapshot_utc_day':when,'scenario':scen,
                                   'algorithm':policy,'budget':k,'replications':len(vals),
                                   'mean_activated':mean,'ci95_lower':max(0.,lo),
                                   'ci95_upper':min(n,hi)})
    save_csv(args.output/'stage9_results.csv',results,
             ['scenario','algorithm','budget','repetitions','mean_activated','ci95_lower','ci95_upper',
              'gain_vs_adaptive_graph','gain_ci95_lower','gain_ci95_upper'])
    save_csv(args.output/'stage9_by_snapshot.csv',detail,
             ['snapshot_utc_day','scenario','algorithm','budget','replications',
              'mean_activated','ci95_lower','ci95_upper'])
    save_csv(args.output/'stage9_context_features.csv',all_features,
             ['snapshot_utc_day','channel','prior_3d_posts','previous_4d_posts',
              'prior_3d_model_score_UNCALIBRATED','time_activity_signal','relative_text_signal'])
    summary_rows=[r for r in results if r['budget']==args.budget]
    image_status=make_plot(args.output/'stage9_comparison.png',results,args.budget)
    summary={
      'stage':9, 'objective':'RETROSPECTIVE synthetic ablation; no observed transmission or psychological effect',
      'dataset':{'proxy_graph_nodes':n,'undirected_content_overlap_edges':len(raw_edges),
                 'channel_day_records':len(signals),'matched_signal_channel_labels':signal_ch_count},
      'design':{'context_dates':CONTEXT_DAYS,'context_feature_window':'previous 7 calendar days ONLY',
                'topology_warning':'Static proxy graph from ALL September; graph is not time-forward, so NO chronological out-of-sample claim.',
                'budget':args.budget,'replications_per_context_scenario':args.repetitions,
                'independent_synthetic_worlds':len(CONTEXT_DAYS)*len(SCENARIOS)*args.repetitions,
                'seed':args.seed,'base_probability_synthetic':args.base_probability,
                'coefficient_magnitude_hypothetical':.60,
                'full_feedback_after_each_seed':True,
                'same_world_across_policies':True,
                'policies':POLICIES,'scenarios':[dict(name=name,time_coefficient=t,text_coefficient=x) for name,t,x in SCENARIOS]},
      'maximum_budget_comparison':summary_rows,
      'runtime_seconds_by_policy_excluding_common_world_generation':
           {p:round(processing_sec[p],4) for p in POLICIES},
      'plot_status':image_status,
      'warnings':[
        'Graph edges represent similar news content, NOT observed follows/reposts/exposure.',
        'All directed arc probabilities and coefficients are imposed, not inferred from Telegram.',
        'The stage5 text model has weak pilot recall and UNCALIBRATED scores: relative descriptors only.',
        'ATSM-like method is a one-step weighted-degree heuristic with complete feedback, NOT a validated optimal algorithm.',
        'An aligned simulation scenario favors use of matching constructed features BY DESIGN; assess graph-only and inverted controls.',
        'The fixed full-month proxy graph contains information from dates later than early context dates; NOT a forward test.',
        '95% CIs are approximate over synthetic worlds, not population or human behavior uncertainty.',
        'Channel-name aliases and two isolated nodes are uncorrected.',
      ],
      'created_utc':datetime.now(timezone.utc).isoformat(),
    }
    (args.output/'stage9_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    combined=[r for r in summary_rows if r['scenario']=='combined_assumption']
    main_numbers='\n'.join(f"- `{r['algorithm']}`: {r['mean_activated']:.3f}; різниця до adaptive_graph_only {r['gain_vs_adaptive_graph']:+.3f} (умовні 95% ДІ [{r['gain_ci95_lower']:.3f}, {r['gain_ci95_upper']:.3f}])." for r in combined)
    report=f'''# Етап 9. Адаптивний алгоритм: часовий і текстовий компоненти (прототип)

## Дані та мета

Використано {n} позначень каналів, {len(raw_edges)} неорієнтованих ребер
**подібності інформаційного змісту**, {len(signals)} статистичних спостережень
«канал–день» із пробної текстової моделі. Перевіряється алгоритмічна гіпотеза,
а **не фактичне поширення** у Telegram і не зміна психологічного стану людей.

## Ознаки

- *Часовий індекс*: зміна частоти публікацій за останні 3 дні порівняно
  з чотирма попередніми (усі дні ДО модельованої дати); обмеження до [-1;1].
- *Текстовий індекс*: відхилення середньої НЕкаліброваної NLP-оцінки каналу
  від глобального середнього в останні 3 дні; шкала .15 умовна, [-1;1].
- *Граф*: фіксований ретроспективний проксі-граф за цілий вересень.
  Він містить відомості з наступних дат для ранніх «зрізів», тому
  це НЕ темпоральна перевірка поза вибіркою.

## Модель та порівняння

Для умовної спрямованої дуги з каналу u до v:

    p(u,v,t)=clip[0.0001, 0.30](0.035 * (0.5+0.9*min(1,overlap/p90))
                               * (1+a*T(u,t)) * (1+b*S(u,t)))

Параметри a,b (0, +0.6 або -0.6) є **заданими синтетичними
сценаріями**, а не встановленими ймовірностями передачі. У напрямі
v→u проводиться окремий незалежний умовний випадковий дослід.

Адаптивна евристика на кожному кроці обирає ще неактивний канал із
найбільшою сумою оцінених прямих ймовірностей активації інших неактивних
каналів. Після кожного вибору виконується повний синтетичний каскад;
наступне рішення враховує його результати. Це адаптивна евристика
на базі зваженого ступеня, **не повний науково валідуваний ATSM-IM**.

Порівняно варіанти з обома ознаками, лише часом, лише текстом, без обох
ознак; статичний MC-greedy і випадковий вибір. Оцінювання на однакових
випадкових світах; 4 часові зрізи × 5 сценаріїв × {args.repetitions} повторів.

## Результат за сценарію з обома умовними залежностями (бюджет {args.budget})

{main_numbers}

## Перевірки проти самопідтвердження

1. `graph_only_control`: модель передачі від сигналів НЕ залежить;
   алгоритм, який використовує сигнали, не повинен отримувати автоматичну перевагу.
2. `inverse_text_control`: умовна текстова залежність змінена на протилежну;
   перевіряємо ризик погіршення при помилковому припущенні.
3. `time_only_assumption` і `text_only_assumption`: окремі абляції.

Вищий результат за `combined_assumption` **не доводить**, що емоційний
текст підвищує реальне охоплення: цю залежність ми навмисно заклали в
модель генерації випадкових світів. Якість текстового класифікатора
за етапом 6 також недостатня для твердження про емпіричний ефект.

## Обмеження

- Лише UTC-день без часу публікації; напрям реальних поширень невідомий.
- Немає спостережуваного графа підписок, пересилань, експозиції або
  верифікованих психологічних реакцій.
- Позначення каналів можуть бути псевдонімами одного каналу.
- 95% інтервали стосуються лише повторних СИНТЕТИЧНИХ світів.
- Для подальших робіт потрібні окреме незалежне оцінювання міток,
  незалежний граф і часовий forward split без витоку майбутньої інформації.
'''
    (args.output/'stage9_method_UA.md').write_text(report,encoding='utf-8')
    print('\nЕтап 9 завершено. Результати:',args.output.resolve(),flush=True)
    print('Надішли stage9_summary.json і stage9_results.csv',flush=True)

if __name__=='__main__':
    main()
