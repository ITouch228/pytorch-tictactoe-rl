import csv
from pathlib import Path


def append_benchmark(cfg, row):
    """Дописывает строку в benchmark.csv — лог прогонов для сравнения версий.

    Заголовок пишется только в пустой/новый файл; старые строки не
    переписываются, поэтому смена набора полей не теряет историю.
    """
    path = Path(cfg.benchmark_path)
    fieldnames = list(row)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
