# Перевірка пакета перед публікацією

Виконано 23.11.2026 в тестовому середовищі Python 3.13:

- `python -m unittest discover -s tests -v` — **18/18 тестів пройдено**.
- `python scripts/run_experiments.py --mode smoke` — **виконано** на графі з 6 вузлів і 5 ребер.
- `python scripts/run_experiments.py --mode real-graph --repetitions 3` — **виконано** на графі з 25 вузлів і 222 ребер.
- `python telegram_stage9_temporal_text_ablation.py --repetitions 3 --planning-worlds 4 --budget 4` — **виконано**.
- Етап 10 у повному обсязі тут **не запускався**: потрібен підготовлений корпус `results_september_stage2/posts_prepared.jsonl`, який не входить до ZIP.

Числові результати пробних запусків на 3 повтореннях **не є** результатами фінального дослідження з 120 повтореннями. У ZIP зберігаються зведення з вихідного архіву.

Авторці слід повторити тестування локально, перевірити скрипти, мітки та ліцензію даних до публікації.
