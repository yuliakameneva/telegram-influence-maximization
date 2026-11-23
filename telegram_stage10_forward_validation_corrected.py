#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Етап 10. ЧАСОВО-КОРЕКТНА перевірка проксі-графа й синтетичних евристик.

Запуск у PythonProject3:
    python telegram_stage10_forward_validation_corrected.py

До кожної контрольної дати алгоритми використовують ТІЛЬКИ попередні 7 днів:
- схожі тексти -> навчальний описовий граф;
- частоту публікацій / некалібровані NLP-бали -> історичні ознаки.
Після фіксації політик окремі 3 НАСТУПНІ дні формують тестовий граф,
який використовується ТІЛЬКИ для оцінювання у штучному Independent Cascade.

Це НЕ вимір реального Telegram-впливу, дифузії або психічних змін.
Моделювання виконується локально; жодних відправлень повідомлень чи API.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

from telegram_stage7_build_proxy_graph import read_posts, match_one_day
from telegram_stage8_simulate_adaptive import live_world, reach, stats, percentile_90

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'results_september_stage2' / 'posts_prepared.jsonl'
DEFAULT_OUT = ROOT / 'results_september_stage10'
SNAPSHOTS = ['2026-09-08','2026-09-15','2026-09-22','2026-09-28']
POLICIES = ['adaptive_temporal_text','adaptive_time_only','adaptive_text_only',
            'adaptive_graph_only','static_past_greedy','random_past']
SCENARIOS = [('future_graph_only',0.0,0.0),
             ('future_aligned_signals',0.6,0.6),
             ('future_inverted_text',0.6,-0.6)]


def fail(msg):
    raise SystemExit('ПОМИЛКА: '+msg)


def csv_write(path, cols, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


# Прозорі маркери СТИЛЮ ТЕКСТУ, а не діагностика емоцій людини.
# Немає навчання на позачасових мітках з етапів 3–6.
EXPRESSIVE_STEMS = (
    'жах', 'страш', 'шок', 'ганьб', 'мерз', 'нелюд', 'горе',
    'неймовір', 'обур', 'радіс', 'термінов', 'срочн', 'ужас',
    'кошмар', 'позор', 'скандал', 'ужас', 'ура',
)
EMOTIVE_EMOJI = tuple('😡🤬😱😭😔🥺🤣🤩🥳😰😨😢😤')
LOUD_WORD = re.compile(r'(?u)\b[А-ЯІЇЄҐA-Z]{4,}\b')


def clamp(v):
    return max(-1.,min(1.,v))


def expressive_style_score(text):
    """Поверхневий експресивний індекс 0..1 для публікації; НЕ емоція."""
    low=text.lower()
    lex=any(stem in low for stem in EXPRESSIVE_STEMS)
    emo=any(symbol in text for symbol in EMOTIVE_EMOJI)
    caps=bool(LOUD_WORD.search(text))
    excl=(text.count('!')+text.count('❗'))>=2
    return min(1., .5*lex+.25*emo+.20*caps+.15*excl)


def build_history_context(train_names, by_day, hist, snapshot):
    """Розрахунок часових і текстових ознак СУВОРО з дат snapshot-7..snapshot-1."""
    recent=hist[-3:]
    previous=hist[:4]
    by_channel_recent=defaultdict(list)
    by_channel_previous=Counter()
    for day in recent:
        assert day < snapshot
        for p in by_day.get(day,()):
            by_channel_recent[p['channel']].append(expressive_style_score(p['text']))
    for day in previous:
        assert day < snapshot
        by_channel_previous.update(p['channel'] for p in by_day.get(day,()))
    all_scores=[v for ch in train_names for v in by_channel_recent[ch]]
    global_score=sum(all_scores)/len(all_scores) if all_scores else 0.
    rows=[]
    for ch in train_names:
        scores=by_channel_recent[ch]
        n_recent=len(scores)
        n_earlier=by_channel_previous[ch]
        recent_rate=n_recent/3.
        earlier_rate=n_earlier/4.
        time_score=clamp((recent_rate-earlier_rate)/max(3.,earlier_rate))
        style_mean=sum(scores)/n_recent if n_recent else 0.
        style_score=clamp((style_mean-global_score)/.15) if n_recent else 0.
        rows.append({
           'snapshot_utc_day':snapshot,'channel':ch,
           'prior_3d_posts':n_recent,'previous_4d_posts':n_earlier,
           'prior_3d_expressive_style_mean':round(style_mean,6) if n_recent else '',
           'global_prior_3d_expressive_style_mean':round(global_score,6),
           'time_activity_signal':round(time_score,6),
           'relative_text_style_signal':round(style_score,6),
        })
    return rows

def get_dates(snapshot, back=7, ahead=3):
    d=date.fromisoformat(snapshot)
    hist=[(d-timedelta(days=i)).isoformat() for i in range(back,0,-1)]
    fut=[(d+timedelta(days=i)).isoformat() for i in range(ahead)]
    assert max(hist)<min(fut), 'Часовий витік'
    return hist,fut


def window_edges(day_matches,day_posts,days,min_matches=1):
    """Тільки задані дні! Граф НЕ містить інформації з-за меж вікна."""
    counts=Counter()
    for day in days:
        counts.update(x['channel'] for x in day_posts.get(day,()))
    agg=defaultdict(lambda: {'matched':0,'days':set()})
    for d in days:
        for (a,b),score,post_a,post_b in day_matches.get(d,()):
            agg[(a,b)]['matched']+=1
            agg[(a,b)]['days'].add(d)
    edges=[]
    for (a,b), info in agg.items():
        if info['matched']<min_matches:continue
        overlap=info['matched']/math.sqrt(counts[a]*counts[b])
        edges.append({'channel_a':a,'channel_b':b,
                      'matched_post_pairs':info['matched'],
                      'shared_days':len(info['days']),
                      'normalized_overlap_rate':round(overlap,8)})
    edges.sort(key=lambda x:(x['channel_a'],x['channel_b']))
    return counts,edges


def edges_to_weighted(edges,index,base_p,q90,signal_time=None,signal_text=None,
                      tcoef=0.,xcoef=0.):
    """Суто синтетичні дуги IC, НЕ оцінені імовірності з Telegram."""
    out=[]
    for r in edges:
        a=index[r['channel_a']]; b=index[r['channel_b']]
        overlap=r['normalized_overlap_rate']
        p=base_p*(.5+.9*min(1.,overlap/q90))
        if signal_time is None:
            out.append((a,b,min(.30,p)))
        else:
            # Джерело може модифікувати обидва напрямки по-різному.
            for src,dst in ((a,b),(b,a)):
                mult=(1+tcoef*signal_time[src])*(1+xcoef*signal_text[src])
                out.append((src,dst,min(.30,max(.0001,p*mult))))
    return out


def undirected_to_arcs(edges):
    return [(a,b,p) for a,b,p in edges for a,b in ((a,b),(b,a))]


def sim_world(n,arcs,rng):
    adj=[0]*n
    for a,b,p in arcs:
        if rng.random()<p:adj[a]|=1<<b
    return adj


def choose_adaptive(n,arcs,active,allowed):
    scores=[0.0]*n
    for a,b,p in arcs:
        if not(active&(1<<a)) and not(active&(1<<b)):
            scores[a]+=p
    cands=[v for v in allowed if not(active&(1<<v))]
    return min(cands,key=lambda v:(-scores[v],v)) if cands else None


def select_static(n,edges,allowed,budget,planning_worlds,seed):
    """Offline greedy ПОВНІСТЮ на минулому графі; evaluation-worlds не доступні."""
    rng=random.Random(seed)
    samples=[]
    for _ in range(planning_worlds):
        # ОДНА спільна реалізація ребер для всіх кандидатів у цьому MC-світі.
        # Це необхідно, щоб прирости seed-наборів були узгодженими в межах світу.
        world = live_world(n, edges, rng)
        samples.append([reach(v, world) for v in allowed])
    selected=[]
    active_masks=[0]*planning_worlds
    for _ in range(budget):
        best,bestval=None,-1
        for loc,v in enumerate(allowed):
            if v in selected:continue
            val=sum((m|samples[j][loc]).bit_count()-m.bit_count()
                    for j,m in enumerate(active_masks))
            if val>bestval:best,bestval=v,val
        if best is None:break
        selected.append(best)
        loc=allowed.index(best)
        active_masks=[m|samples[j][loc] for j,m in enumerate(active_masks)]
    return selected


def eval_adaptive(n,world,assumed_arcs,allowed,budget):
    active=0
    for _ in range(budget):
        nxt=choose_adaptive(n,assumed_arcs,active,allowed)
        if nxt is not None:active|=reach(nxt,world,active)
    return active.bit_count()


def eval_static(world,seeds,budget):
    active=0
    for v in seeds[:budget]:
        active|=reach(v,world,active)
    return active.bit_count()


def make_plot(path,rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'no_matplotlib'
    data=[r for r in rows if r['scenario']=='future_graph_only']
    names=[p for p in POLICIES]
    means=[next(r['mean_activated'] for r in data if r['algorithm']==p) for p in names]
    fig,ax=plt.subplots(figsize=(11,4.6))
    ax.bar(range(len(names)),means)
    ax.set_xticks(range(len(names)),names,rotation=25,ha='right',fontsize=8)
    ax.set_ylim(0,max(1.,max(means))*1.22)
    ax.set_ylabel('Synthetic activated nodes')
    ax.set_title('Stage 10: time-forward graph holdout, no future information for policy')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    return 'saved'


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    a.add_argument('--output',type=Path,default=DEFAULT_OUT)
    a.add_argument('--repetitions',type=int,default=120)
    a.add_argument('--planning-worlds',type=int,default=180)
    a.add_argument('--budget',type=int,default=4)
    a.add_argument('--base-p',type=float,default=.035)
    a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--snapshot-days',nargs='+',default=SNAPSHOTS)
    args=a.parse_args()
    if args.repetitions<1 or args.planning_worlds<1 or not(1<=args.budget<=25) or not(0<args.base_p<=.10):
        a.error('Некоректний бюджет, повторення або ймовірність')
    snaps=sorted(set(args.snapshot_days))
    for dt in snaps:
        try: d=date.fromisoformat(dt)
        except ValueError:fail('Некоректна контрольна дата '+dt)
        if not(date(2026,9,8)<=d<=date(2026,9,28)):
            fail('Дата повинна бути в межах 08–28 вересня (для 7 минулих і 3 майбутніх днів)')
    by_day,all_counts,counts=read_posts(args.input,40,1400)
    if len(all_counts)<3:fail('Замало назв каналів для порівняння')
    day_matches={}
    relevant=set()
    for s in snaps:
        h,f=get_dates(s); relevant.update(h+f)
    print('Етап 10. Розрахунок текстових збігів за днями (без перетину вікон)...',flush=True)
    for i,dt in enumerate(sorted(relevant),1):
        matched,ncand=match_one_day(by_day.get(dt,[]),.70,.50,3,35,25000)
        day_matches[dt]=matched
        print(f'  [{i}/{len(relevant)}] {dt}: {len(matched)} зіставлень',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    allvals=defaultdict(list)
    bysnap=defaultdict(list)
    overview=[]; history_rows=[]; future_rows=[]; contexts=[]; first_choices=[]
    runtime=Counter();checks=[]
    for ix,snap in enumerate(snaps):
        hist,future=get_dates(snap)
        assert len(set(hist)&set(future))==0
        past_counts,past_edges=window_edges(day_matches,by_day,hist)
        future_counts,future_edges=window_edges(day_matches,by_day,future)
        train_names=sorted(past_counts)
        future_names=sorted(future_counts)
        # Індекси ДОСТУПНИХ вузлів залежать лише від минулого.
        # Нові майбутні назви додаємо в кінець лише для тестового середовища.
        union=train_names+sorted(set(future_names)-set(train_names))
        if not train_names:fail('Немає каналів до '+snap)
        if len(train_names)<args.budget:fail('У попередньому вікні замало каналів до '+snap)
        if len(union)>63:fail('Завеликий граф для компактного моделювання')
        n=len(union);idx={name:i for i,name in enumerate(union)}
        allowed=[idx[x] for x in train_names]
        past_q=percentile_90([v['normalized_overlap_rate'] for v in past_edges]) if past_edges else .05
        future_q=percentile_90([v['normalized_overlap_rate'] for v in future_edges]) if future_edges else .05
        # Ознаки зі СТИЛЮ ПУБЛІКАЦІЙ і активності: тільки t-7 ... t-1.
        # Не беремо stage5 model scores: той класифікатор навчався на всьому вересні.
        rows=build_history_context(train_names,by_day,hist,snap)
        assert all((r['snapshot_utc_day']==snap) for r in rows)
        contexts.extend(rows)
        time_sig=[0.]*n;text_sig=[0.]*n
        for r in rows:
            j=idx[r['channel']]
            time_sig[j]=float(r['time_activity_signal'])
            text_sig[j]=float(r['relative_text_style_signal'])
        past_base=edges_to_weighted(past_edges,idx,args.base_p,past_q)
        assumed={
          'adaptive_graph_only':undirected_to_arcs(past_base),
          'adaptive_time_only':edges_to_weighted(past_edges,idx,args.base_p,past_q,time_sig,text_sig,.6,0.),
          'adaptive_text_only':edges_to_weighted(past_edges,idx,args.base_p,past_q,time_sig,text_sig,0.,.6),
          'adaptive_temporal_text':edges_to_weighted(past_edges,idx,args.base_p,past_q,time_sig,text_sig,.6,.6),
        }
        plan=select_static(n,past_base,allowed,args.budget,args.planning_worlds,args.seed+ix*10007)
        for policy in POLICIES:
            if policy=='random_past':
                continue  # Випадкові початкові вузли варіюють між симуляціями.
            first=plan[0] if policy=='static_past_greedy' else choose_adaptive(n,assumed[policy],0,allowed)
            first_choices.append({'snapshot_utc_day':snap,'algorithm':policy,
                                  'first_selected_channel_from_history':union[first] if first is not None else ''})
        # Майбутній граф є ЛИШЕ середовищем оцінювання — його ребра та q90
        # не передаються жодній із шести політик планування/вибору.
        fbase=edges_to_weighted(future_edges,idx,args.base_p,future_q)
        fpairs={(r['channel_a'],r['channel_b']) for r in future_edges}
        hpairs={(r['channel_a'],r['channel_b']) for r in past_edges}
        common=len(fpairs&hpairs)
        overview.append({
          'snapshot_utc_day':snap,'history_first_day':hist[0],'history_last_day':hist[-1],
          'test_first_day':future[0],'test_last_day':future[-1],
          'past_channels':len(train_names),'future_channels':len(future_names),
          'past_eligible_posts':sum(past_counts.values()),
          'future_eligible_posts':sum(future_counts.values()),
          'new_future_channels_unselectable':len(set(future_names)-set(train_names)),
          'past_edges':len(hpairs),'future_edges':len(fpairs),'intersection_edges':common,
          'history_edge_precision_for_future':round(common/len(hpairs),6) if hpairs else None,
          'future_edge_recall_from_history':round(common/len(fpairs),6) if fpairs else None,
          'edge_jaccard':round(common/len(hpairs|fpairs),6) if hpairs|fpairs else None,
        })
        for e in past_edges:history_rows.append({'snapshot_utc_day':snap,**e})
        for e in future_edges:future_rows.append({'snapshot_utc_day':snap,**e})
        checks.append({'snapshot_utc_day':snap,
                       'max_available_date_for_policy':hist[-1],
                       'first_evaluation_date':future[0],
                       'history_future_date_disjoint':True,
                       'past_graph_only_from_history':True,
                       'context_only_uses_dates_before_snapshot':True,
                       'context_uses_train_free_past_only_style_markers':True,
                       'new_future_nodes_not_selectable':True,
                       'future_proxy_graph_exclusive_to_evaluator':True})
        for sc_ix,(scenario,tcoef,xcoef) in enumerate(SCENARIOS):
            evaluation_arcs=edges_to_weighted(future_edges,idx,args.base_p,future_q,time_sig,text_sig,tcoef,xcoef)
            if tcoef==0 and xcoef==0:
                evaluation_arcs=undirected_to_arcs(fbase)
            for rep in range(args.repetitions):
                world=sim_world(n,evaluation_arcs,
                               random.Random(args.seed+ix*1_000_000+sc_ix*100_000+rep))
                seq=allowed[:]
                random.Random(args.seed+7_000_000+ix*100_000+sc_ix*10_000+rep).shuffle(seq)
                for policy in POLICIES:
                    start=time.perf_counter()
                    if policy=='static_past_greedy':
                        y=eval_static(world,plan,args.budget)
                    elif policy=='random_past':
                        y=eval_static(world,seq,args.budget)
                    else:
                        y=eval_adaptive(n,world,assumed[policy],allowed,args.budget)
                    runtime[policy]+=time.perf_counter()-start
                    allvals[(scenario,policy)].append(y)
                    bysnap[(snap,scenario,policy)].append(y)
        print(f'  Контрольна дата {snap}: {len(past_edges)} минулих / {len(future_edges)} майбутніх ребер; {len(train_names)} доступних каналів',flush=True)
    results=[]; per_snapshot=[]
    for sc,_,_ in SCENARIOS:
        base=allvals[(sc,'adaptive_graph_only')]
        for pol in POLICIES:
            v=allvals[(sc,pol)]
            mu,lo,hi=stats(v)
            dm,dlo,dhi=stats([x-y for x,y in zip(v,base)])
            results.append({'scenario':sc,'algorithm':pol,'budget':args.budget,
                            'repetitions':len(v),'mean_activated':mu,
                            'ci95_lower':lo,'ci95_upper':hi,
                            'gain_vs_past_graph_adaptive':dm,
                            'gain_ci95_lower':dlo,'gain_ci95_upper':dhi})
    for snap in snaps:
        for sc,_,_ in SCENARIOS:
            base=bysnap[(snap,sc,'adaptive_graph_only')]
            for pol in POLICIES:
                v=bysnap[(snap,sc,pol)]
                mu,lo,hi=stats(v)
                dm,dlo,dhi=stats([x-y for x,y in zip(v,base)])
                per_snapshot.append({'snapshot_utc_day':snap,'scenario':sc,'algorithm':pol,
                                     'budget':args.budget,'replications':len(v),
                                     'mean_activated':mu,'ci95_lower':lo,'ci95_upper':hi,
                                     'gain_vs_past_graph_adaptive':dm,
                                     'gain_ci95_lower':dlo,'gain_ci95_upper':dhi})
    csv_write(args.output/'stage10_results.csv',list(results[0]),results)
    csv_write(args.output/'stage10_by_snapshot.csv',list(per_snapshot[0]),per_snapshot)
    csv_write(args.output/'stage10_graph_stability.csv',list(overview[0]),overview)
    cols=['snapshot_utc_day','channel_a','channel_b','matched_post_pairs','shared_days','normalized_overlap_rate']
    csv_write(args.output/'stage10_history_edges.csv',cols,history_rows)
    csv_write(args.output/'stage10_future_edges_EVAL_ONLY.csv',cols,future_rows)
    csv_write(args.output/'stage10_context_features.csv',
              ['snapshot_utc_day','channel','prior_3d_posts','previous_4d_posts','prior_3d_expressive_style_mean','global_prior_3d_expressive_style_mean','time_activity_signal','relative_text_style_signal'],contexts)
    csv_write(args.output/'stage10_first_choices_PAST_ONLY.csv',
              ['snapshot_utc_day','algorithm','first_selected_channel_from_history'],first_choices)
    (args.output/'stage10_leakage_checks.json').write_text(
       json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    pic=make_plot(args.output/'stage10_comparison.png',results)
    summary={
      'stage':10,'purpose':'Time-forward evaluation on NEXT 3-DAY CONTENT-SIMILARITY GRAPH; synthetic IC only',
      'input':str(args.input),'context_signal_source':'past-only post text and post counts; NO stage5 NLP predictions',
      'posts_in_scope':counts.get('in_scope',0),'channel_labels_seen_all_month_DIAGNOSTIC_ONLY':len(all_counts),
      'fixed_history_days':7,'future_graph_evaluation_days':3,
      'snapshots':overview,'budget':args.budget,'repetitions_per_snapshot_scenario':args.repetitions,
      'total_worlds':len(snaps)*len(SCENARIOS)*args.repetitions,
      'policies':POLICIES,'synthetic_scenarios':[x[0] for x in SCENARIOS],
      'probability_design':'future text-overlap edges evaluated using synthetic Bernoulli IC, base_p=%.5f and hypothetical optional coefficients +/-0.6'%args.base_p,
      'summary_comparison':results,'seconds_by_algorithm_excluding_graph_build_and_world_generation':dict(runtime),
      'plot_status':pic,
      'leakage_guard':'Policy graph, available actions, textual scores and time features all strictly prior to snapshot; held-out future graph used only for evaluation.',
      'key_limitations':[
        'Future graph reflects future same-day TEXTUAL overlap, not Telegram reposts, social ties or human exposure.',
        'Future graph may be empty/sparse; missing nodes cannot be chosen if absent from the past.',
        'IC probabilities and coefficients are synthetic hypotheses, not measured from Telegram.',
        'The feature-aligned synthetic scenario is favorable to feature-based policies by construction; graph-only scenario is primary.',
        'Text signals in this experiment are rule-based expressive STYLE markers from prior days, not stage5 NLP predictions or human emotions.',
        'This is a sequential time-forward evaluation over 4 dates in ONE month, not long-term generalization.',
        'Name-label aliases and TF-IDF false matches remain possible; link stability is not causal diffusion.',
        'All confidence intervals reflect only Monte Carlo randomness, not dataset sampling or psychological response uncertainty.',
        'Post-level source timestamp has UTC-day resolution only, so within-day causal ordering cannot be verified.',
      ],
      'generated_utc':datetime.now(timezone.utc).isoformat(),
    }
    (args.output/'stage10_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# Етап 10. Часово-коректний контрольний експеримент

**Мета.** Перевірити сталість тематичного проксі-графа та синтетичних
політик адаптивного вибору за умови, що **на дату рішення доступні лише
попередні повідомлення**.

## Дизайн

Для кожної контрольної дати побудовано граф із **попередніх 7 днів**.
Часовий сигнал активності й текстовий сигнал за попередні 7 днів беруться
лише з історичних публікацій. Текстовий сигнал — прозорий індекс
експресивних лексем, емодзі та графічного оформлення; він НЕ є
прогнозом NLP-моделі з етапу 5, бо та навчалася на повідомленнях
усього місяця та спричинила б опосередкований витік майбутнього. Окремо побудовано **майбутній проксі-граф
за наступні 3 календарні дні**. Умови відбору близьких текстів відповідають
етапу 7 (добовий TF-IDF, косинусна подібність не нижча 0,70, лексичне
перекриття не нижче 0,50), але задля коротких вікон ребро існує вже за
одного збігу (не за трьох у двох днях, як у загальному графі етапу 7).

Навчальний та майбутній графи мають **неперетинні дати**. У політики
передаються лише попередні ребра та ознаки. Канали, яких немає у
навчальному вікні, недоступні для вибору. Майбутній граф використано
лише для оцінювання результату на умовних незалежних каскадах.

## Метрики

Перетин/Жаккарова схожість навчальних і майбутніх ребер; кількість
змодельованих активованих вузлів; парні різниці до графового адаптивного
методу; наближені 95% інтервали у повтореннях Монте-Карло.

## Контрольні сценарії

- `future_graph_only` — умовна передача залежить лише від майбутнього
  графа схожих публікацій; **основна перевірка**, без підлаштування
  до історичних текстових/часових сигналів.
- `future_aligned_signals` — штучні ймовірності додатково залежать від
  історичних сигналів: **перевага відповідних методів тут можлива за дизайном**.
- `future_inverted_text` — умовна залежність від тексту зворотна.

## Обмеження

Цей контроль НЕ є точним повторенням етапу 9: тут інший,
навчально-незалежний індикатор стилю тексту задля часової чистоти.

Це **не** прогноз справжніх пересилань і **не** вимір
інформаційно-психологічного впливу. Топологія майбутніх збігів текстів
також не є справжнім каналом передачі інформації. Коефіцієнти моделі
передачі не оцінювалися з даних. Часові межі знімають витік майбутньої
інформації в рішення, але не усувають усіх зміщень даних або змін назв.

Для написання практичної частини використовуй отримані CSV, JSON та PNG.
Зазначай, що це **синтетичний часово-коректний контрольний експеримент**.
'''
    (args.output/'stage10_method_UA.md').write_text(report,encoding='utf-8')
    print('\nЕтап 10 завершено! Папка:',args.output.resolve())
    print('Надішли stage10_summary.json, stage10_results.csv і stage10_graph_stability.csv')

if __name__=='__main__':
    main()
