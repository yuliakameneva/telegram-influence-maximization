#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Етап 8: ВІДТВОРЮВАНЕ СИНТЕТИЧНЕ порівняння алгоритмів на проксі-графі.

Запуск: python telegram_stage8_simulate_adaptive.py
Бібліотеки: стандартна бібліотека Python (matplotlib лише для графіка).

ВАЖЛИВО. Ребра етапу 7 означають подібність текстів, а НЕ реальне
пересилання чи вплив. Імовірності передачі інформації УМОВНІ, а не
виміряні на Telegram. Ніякі реальні пости не публікуються та не надсилаються.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_EDGE_PATH = ROOT / 'results_september_stage7' / 'stage7_proxy_edges.csv'
DEFAULT_NODE_PATH = ROOT / 'results_september_stage7' / 'stage7_channel_nodes.csv'
DEFAULT_OUTPUT = ROOT / 'results_september_stage8'
SEED = 2026
POLICIES = ['adaptive_mc', 'adaptive_degree', 'static_greedy', 'degree', 'random']
SCENARIOS = [
    {'name': 'low', 'base_p': 0.015, 'min_overlap': 0.0, 'recency': False},
    {'name': 'middle', 'base_p': 0.035, 'min_overlap': 0.0, 'recency': False},
    {'name': 'high', 'base_p': 0.060, 'min_overlap': 0.0, 'recency': False},
    {'name': 'middle_sparse', 'base_p': 0.035, 'min_overlap': 0.025, 'recency': False},
    {'name': 'middle_recency', 'base_p': 0.035, 'min_overlap': 0.0, 'recency': True},
]


def fail(message):
    raise SystemExit('ПОМИЛКА: ' + message)


def read_graph(edge_path: Path, node_path: Path):
    if not node_path.exists():
        fail(f'Немає {node_path}. Потрібно спочатку завершити етап 7.')
    if not edge_path.exists():
        fail(f'Немає {edge_path}. Потрібно спочатку завершити етап 7.')
    with node_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'channel', 'eligible_post_count'}.issubset(set(reader.fieldnames or [])):
            fail('У stage7_channel_nodes.csv неправильні назви колонок.')
        node_rows = list(reader)
    names = [r['channel'].strip() for r in node_rows]
    if len(names) != len(set(names)) or len(names) < 3 or any(not v for v in names):
        fail('Некоректні або повторені позначення вузлів у stage7_channel_nodes.csv.')
    names.sort()
    index = {name: i for i, name in enumerate(names)}
    with edge_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'channel_a', 'channel_b', 'matched_post_pairs', 'shared_days', 'normalized_overlap_rate', 'last_matched_day_utc'}
        if not needed.issubset(set(reader.fieldnames or [])):
            fail('У stage7_proxy_edges.csv відсутні необхідні колонки. Бери файл саме з results_september_stage7.')
        rows = list(reader)
    if not rows:
        fail('Немає ребер у stage7_proxy_edges.csv.')
    edges = []
    seen = set()
    try:
        for r in rows:
            a, b = index[r['channel_a']], index[r['channel_b']]
            if a == b:
                fail('У графі виявлено петлю вузла.')
            pair = (min(a,b), max(a,b))
            if pair in seen:
                fail('У stage7_proxy_edges.csv повторено одну й ту саму пару каналів.')
            seen.add(pair)
            overlap = float(r['normalized_overlap_rate'])
            matched, days = int(r['matched_post_pairs']), int(r['shared_days'])
            last = date.fromisoformat(r['last_matched_day_utc'])
            if not (0 <= overlap <= 1 and 1 <= days <= 31 and matched >= 1):
                fail('Некоректні значення показників ребра.')
            if not date(2026, 9, 1) <= last <= date(2026, 9, 30):
                fail('Експеримент розрахований на вересневий корпус 2026.')
            edges.append((pair[0],pair[1],overlap,days,last))
    except (ValueError, KeyError) as exc:
        fail(f'Не вдалося прочитати ребра: {exc}')
    n = len(names)
    if n > 63:
        fail('Цю компактну версію експерименту перевірено для графів не більш ніж із 63 вузлами.')
    return n, edges


def percentile_90(values):
    vals = sorted(values)
    idx = math.ceil(0.9 * len(vals)) - 1
    return max(1e-9, vals[max(idx, 0)])


def graph_scenario(n, edges, q90, scenario):
    final = []
    for a,b,overlap,shared_days,last in edges:
        if overlap < scenario['min_overlap']:
            continue
        # Умовне зіставлення сили тематичного перекриття з параметром
        # математичної симуляції. НЕ емпірична ймовірність пересилання.
        strength = min(1.0, overlap / q90)
        factor = 0.5 + 0.9 * strength
        freshness = 1.0
        if scenario['recency']:
            age = (date(2026,9,30) - last).days
            freshness = 0.55 + 0.45 * 2.0 ** (-age / 7.0)
        p = min(0.30, scenario['base_p'] * factor * freshness)
        final.append((a,b,p))
    return final


def live_world(n, edges, rng):
    """Незалежно визначаємо активність спрямованих дуг в обидва боки."""
    adjacency = [0]*n
    for a,b,p in edges:
        if rng.random() < p:
            adjacency[a] |= 1<<b
        if rng.random() < p:
            adjacency[b] |= 1<<a
    return adjacency


def reach(src, adjacency, blocked=0):
    """Нова активована множина після повного завершення каскаду."""
    bit = 1 << src
    if blocked & bit:
        return 0
    reached = bit
    queue = [src]
    while queue:
        src = queue.pop()
        candidates = adjacency[src] & ~blocked & ~reached
        while candidates:
            nxt = candidates & -candidates
            candidates -= nxt
            reached |= nxt
            queue.append(nxt.bit_length()-1)
    return reached


def static_mc_plan(n, edges, planning_worlds, rng, kmax):
    """Звичайний offline greedy до початку будь-якого каскаду."""
    sample = []
    for _ in range(planning_worlds):
        world = live_world(n,edges,rng)
        sample.append([reach(i,world) for i in range(n)])
    accum = [0]*planning_worlds
    selected = []
    for _ in range(min(n, kmax)):
        best, best_score = None, -1
        for i in range(n):
            if i in selected:
                continue
            score = sum((m | sample[j][i]).bit_count() - m.bit_count()
                        for j,m in enumerate(accum))
            if score > best_score:
                best_score, best = score, i
        selected.append(best)
        for j in range(planning_worlds):
            accum[j] |= sample[j][best]
    return selected


def degree_plan(n, edges):
    weighted = [0.0]*n
    for a,b,p in edges:
        weighted[a] += p
        weighted[b] += p
    return sorted(range(n),key=lambda v:(-weighted[v],v))


def adaptive_mc_pick(n, edges, active_mask, inner_worlds, rng):
    """Оцінка умовного приросту серед ще НЕ активних вузлів.

    Повний зворотний зв'язок: уже активована множина відома, каскад завершився.
    У незалежній live-edge IC моделі ребра між неактивними вузлами й надалі
    незалежні; невдалі переходи від активних до неактивних вже відомі.
    """
    candidates = [v for v in range(n) if not (active_mask & (1<<v))]
    if not candidates:
        return None
    gains = [0]*n
    for _ in range(inner_worlds):
        sample = live_world(n,edges,rng)
        for v in candidates:
            gains[v] += reach(v,sample,active_mask).bit_count()
    return min(candidates,key=lambda v:(-gains[v],v))


def adaptive_degree_pick(n, edges, active_mask):
    gains = [0.0]*n
    for a,b,p in edges:
        if not (active_mask & (1<<a)) and not (active_mask & (1<<b)):
            gains[a] += p
            gains[b] += p
    candidates = [v for v in range(n) if not active_mask & (1<<v)]
    if not candidates:
        return None
    return min(candidates,key=lambda v:(-gains[v],v))


def standard_evaluate(world, seq, kmax):
    active = 0
    vals = []
    for step in range(kmax):
        v = seq[step]
        active |= reach(v,world,active)
        vals.append(active.bit_count())
    return vals


def adaptive_evaluate(world, n, edges, kmax, inner_worlds, rng, kind):
    active = 0
    vals = []
    for _ in range(kmax):
        if kind == 'adaptive_mc':
            v = adaptive_mc_pick(n,edges,active,inner_worlds,rng)
        else:
            v = adaptive_degree_pick(n,edges,active)
        if v is not None:
            active |= reach(v,world,active)
        vals.append(active.bit_count())
    return vals


def stats(values):
    n = len(values)
    avg = statistics.fmean(values)
    se = statistics.stdev(values)/(n**0.5) if n > 1 else 0.0
    return round(avg,5), round(avg-1.96*se,5), round(avg+1.96*se,5)


def save_csv(path, rows, cols):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out, records):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'skipped_no_matplotlib'
    selected = [r for r in records if r['scenario']=='middle']
    fig,ax=plt.subplots(figsize=(9.2,5.5))
    for policy in POLICIES:
        rs = [r for r in selected if r['algorithm']==policy]
        ax.plot([r['budget'] for r in rs], [r['mean_activated'] for r in rs],marker='o',label=policy)
    ax.set(title='Synthetic Independent Cascade: comparison of policies (middle)',
           xlabel='Number of initial activations (budget)',ylabel='Mean activated nodes')
    ax.set_ylim(0,max(r['mean_activated'] for r in selected)*1.13)
    ax.set_xticks(sorted(set(r['budget'] for r in selected)))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'stage8_comparison_middle.png',dpi=160)
    plt.close(fig)
    return 'saved'


def main():
    parser=argparse.ArgumentParser(description='Етап 8 — синтетична симуляція адаптивного покриття проксі-графа')
    parser.add_argument('--edges',type=Path,default=DEFAULT_EDGE_PATH)
    parser.add_argument('--nodes',type=Path,default=DEFAULT_NODE_PATH)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--repetitions',type=int,default=120)
    parser.add_argument('--planning-worlds',type=int,default=180)
    parser.add_argument('--adaptive-worlds',type=int,default=24)
    parser.add_argument('--budget',type=int,default=4)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    if min(args.repetitions,args.planning_worlds,args.adaptive_worlds,args.budget)<1:
        parser.error('Усі числові параметри повинні бути додатними.')
    n, edges=read_graph(args.edges,args.nodes)
    if args.budget>n:
        parser.error('Бюджет не може перевищувати кількість вузлів.')
    q90=percentile_90([e[2] for e in edges])
    density=round(2*len(edges)/(n*(n-1)),5)
    args.output.mkdir(parents=True,exist_ok=True)
    print(f'Етап 8: {n} позначень каналів, {len(edges)} ребер; щільність {density}',flush=True)
    print(f'Симуляцій на сценарій: {args.repetitions}; бюджет: {args.budget}',flush=True)
    outcome=[]
    scenario_descriptions=[]
    compute_costs=[]
    for sidx,s in enumerate(SCENARIOS):
        weighted=graph_scenario(n,edges,q90,s)
        # Незалежне від evaluation-world планування базового алгоритму.
        static_rng=random.Random(args.seed + 101_000+sidx*1000)
        planned_seconds={name:0.0 for name in POLICIES}
        eval_seconds={name:0.0 for name in POLICIES}
        t0=time.perf_counter()
        static_seq=static_mc_plan(n,weighted,args.planning_worlds,static_rng,args.budget)
        planned_seconds['static_greedy']=time.perf_counter()-t0
        t0=time.perf_counter()
        degree_seq=degree_plan(n,weighted)[:args.budget]
        planned_seconds['degree']=time.perf_counter()-t0
        scores={p:[[] for _ in range(args.budget)] for p in POLICIES}
        # ОДНАКОВІ реалізації мережі для всіх порівнюваних методів.
        for rep in range(args.repetitions):
            world=live_world(n,weighted,random.Random(args.seed+10_000*sidx+rep))
            random_seq=list(range(n))
            random.Random(args.seed+90_000+sidx*5000+rep).shuffle(random_seq)
            # Час генерації випадкового світу спільний для всіх методів
            # і не включається до їхніх обчислювальних витрат.
            for algo in POLICIES:
                t0=time.perf_counter()
                if algo=='static_greedy':
                    values=standard_evaluate(world,static_seq,args.budget)
                elif algo=='degree':
                    values=standard_evaluate(world,degree_seq,args.budget)
                elif algo=='random':
                    values=standard_evaluate(world,random_seq,args.budget)
                elif algo=='adaptive_degree':
                    values=adaptive_evaluate(world,n,weighted,args.budget,
                                             args.adaptive_worlds,None,'adaptive_degree')
                else:
                    rng=random.Random(args.seed+5_000_000+sidx*100_000+rep)
                    values=adaptive_evaluate(world,n,weighted,args.budget,
                                             args.adaptive_worlds,rng,'adaptive_mc')
                eval_seconds[algo]+=time.perf_counter()-t0
                for idx,v in enumerate(values):
                    scores[algo][idx].append(v)
        for budget in range(1,args.budget+1):
            reference=scores['static_greedy'][budget-1]
            for algo in POLICIES:
                vals=scores[algo][budget-1]
                mean,lo,hi=stats(vals)
                diffs=[x-y for x,y in zip(vals,reference)]
                dm,dlo,dhi=stats(diffs)
                rec={'scenario':s['name'],'algorithm':algo,'budget':budget,
                     'replications':args.repetitions,'mean_activated':mean,
                     'ci95_lower':max(0,lo),'ci95_upper':min(n,hi),
                     'fraction_activated':round(mean/n,5),
                     'mean_gain_vs_static_greedy':dm,
                     'gain_ci95_lower':dlo,'gain_ci95_upper':dhi}
                outcome.append(rec)
        for algo in POLICIES:
            compute_costs.append({'scenario':s['name'],'algorithm':algo,
                 'offline_planning_seconds':round(planned_seconds[algo],5),
                 'evaluation_and_decisions_seconds':round(eval_seconds[algo],5),
                 'total_excluding_common_world_generation_seconds':round(planned_seconds[algo]+eval_seconds[algo],5),
                 'replications':args.repetitions})
        scenario_descriptions.append({'name':s['name'],'base_synthetic_probability':s['base_p'],
           'min_proxy_overlap':s['min_overlap'],'recency_factor_enabled':s['recency'],
           'retained_undirected_edges':len(weighted),'directed_trials_per_world':2*len(weighted),
           'mean_synthetic_arc_probability':round(statistics.fmean([e[2] for e in weighted]),5)})
        mid = [r for r in outcome if r['scenario']==s['name'] and r['budget']==args.budget]
        print(f"[{sidx+1}/{len(SCENARIOS)}] {s['name']} ({len(weighted)} ребер): "+
              ', '.join(f"{r['algorithm']}={r['mean_activated']:.2f}" for r in mid),flush=True)
    columns=['scenario','algorithm','budget','replications','mean_activated',
      'ci95_lower','ci95_upper','fraction_activated','mean_gain_vs_static_greedy',
      'gain_ci95_lower','gain_ci95_upper']
    save_csv(args.output/'stage8_results.csv',outcome,columns)
    save_csv(args.output/'stage8_sensitivity.csv',
      [r for r in outcome if r['budget']==args.budget],columns)
    save_csv(args.output/'stage8_compute_cost.csv',compute_costs,
       ['scenario','algorithm','offline_planning_seconds','evaluation_and_decisions_seconds',
        'total_excluding_common_world_generation_seconds','replications'])
    chart=plot_results(args.output,outcome)
    summary={
      'stage':8,
      'objective':'SYNTHETIC Monte Carlo Independent Cascade comparison on a CONTENT SIMILARITY PROXY graph',
      'graph':{'nodes_as_channel_labels':n,'proxy_edges':len(edges),'undirected_density':density,
               'overlap_rate_p90':round(q90,6),'potential_aliases_not_merged':True},
      'config':{'seed':args.seed,'replications_per_scenario':args.repetitions,
                'offline_planning_mc_worlds':args.planning_worlds,
                'online_adaptive_mc_worlds_per_decision':args.adaptive_worlds,
                'max_seed_budget':args.budget,
                'feedback':'full cascade completion after every addition',
                'randomness':'paired evaluation worlds across all policies in each replication',
                'arc_trials':'independent directed Bernoulli trials in both directions for each undirected proxy edge',
                'probability_design':'synthetic p=base_p*(0.5+0.9*min(1, normalized_overlap/p90))*optional_recency; cap .30'},
      'scenarios':scenario_descriptions,
      'middle_scenario_at_max_budget':[r for r in outcome if r['scenario']=='middle' and r['budget']==args.budget],
      'chart_status':chart,
      'middle_scenario_compute_costs':[r for r in compute_costs if r['scenario']=='middle'],
      'limitations':[
        'This is not observed Telegram diffusion, forwarding, readership, or human psychological change.',
        'Edge weights are freely specified sensitivity scenarios and NOT estimated transmission probabilities.',
        'Only topology from similar content was used. No textual emotion or audience response included.',
        'This is a prototype of the adaptive selection component, not a validated full ATSM-IM implementation.',
        'The unusually dense graph and possible channel aliases affect the simulation; compare sparse and low/high regimes.',
        'CI intervals refer to repeated draws of the SYNTHETIC world, not real-world uncertainty.',
        'Wall-clock compute costs are machine-dependent, exclude common world generation, and include offline planning where applicable.',
        'Comparisons share evaluation-world draws; planning simulations are independent of evaluation worlds.',
        'Source edges do not provide causal direction or within-day publication sequence.',
      ],
      'generated_utc':datetime.now(timezone.utc).isoformat(),
    }
    (args.output/'stage8_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    best={r['algorithm']:r['mean_activated'] for r in summary['middle_scenario_at_max_budget']}
    report=f'''# Етап 8. Синтетична експериментальна симуляція

## Постановка

На графі **тематичної подібності контенту** з {n} позначеннями каналів та {len(edges)}
неорієнтованими ребрами перевірено **гіпотетичну** модель Independent Cascade (IC).
Це не дані про фактичний інформаційно-психологічний вплив чи пересилання.

Для кожного ребра моделюються дві НЕЗАЛЕЖНІ орієнтовані передачі з умовною
ймовірністю p. Невідомі реальні ймовірності замінено на задані сценарії
0,015 / 0,035 / 0,060, з перетворенням сили тематичного перекриття, окремо
перевірено розрідження графа та умовний коефіцієнт давності збігів.

## Алгоритми

1. `adaptive_mc`: послідовне додавання вузла за максимальним очікуваним
   додатковим покриттям (Monte Carlo), із повним спостереженням результату
   попереднього каскаду. Це ЕКСПЕРИМЕНТАЛЬНИЙ адаптивний компонент.
2. `static_greedy`: попереднє жадібне планування набору вузлів за MC без
   спостереження проміжних результатів.
3. `adaptive_degree`: послідовний вибір за залишковим зваженим ступенем.
4. `degree`: попередній вибір вузлів за зваженим ступенем.
5. `random`: випадковий попередній вибір.

## Дизайн експерименту

- Фіксований seed: {args.seed}; повторень на сценарій: {args.repetitions}.
- Однаковий бюджет для всіх методів: 1...{args.budget} умовних активацій.
- Для кожного повторення одна і та сама випадково згенерована модель мережі
  оцінює кожен метод; планувальні випадкові реалізації відокремлені.
- Повне завершення кожного каскаду до нового рішення адаптивного методу.
- Звітуються середні кількості активованих вузлів та приблизні 95% інтервали
  лише за синтетичною випадковістю. Порівняння з `static_greedy` є парним.

## Результат середнього сценарію за максимального бюджету

'''+''.join(f'- `{name}`: {val:.3f} активованих вузлів у середньому.\n' for name,val in best.items())+'''
## Обмеження та наступні кроки

Граф не є реальною мережею впливу; вплив і впливовість каналів не виміряні.
Коефіцієнт часу тут — аналіз чутливості параметризації ребер, а не напрям чи
фактичний порядок передачі в Telegram. Через недостатню якість пілотної
емоційної класифікації емоційні параметри в дифузію НЕ підставлялися.
Це демонстрація та порівняння **адаптивного компоненту**, а НЕ повна
емпірично підтверджена реалізація ATSM-IM.

Надалі: інтерпретація результатів, аналіз переваг / відсутності переваг
адаптивного рішення, абляції, текст для ІНДЗ, за можливості інший граф
для перевірки стійкості висновків.
'''
    (args.output/'stage8_method_UA.md').write_text(report,encoding='utf-8')
    print('\nЕтап 8 завершено! Файли у:',args.output.resolve(),flush=True)
    print('Надішли stage8_summary.json та stage8_results.csv',flush=True)

if __name__=='__main__':
    main()
