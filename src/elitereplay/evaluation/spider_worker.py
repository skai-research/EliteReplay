"""Resource-bounded read-only SQL execution with pinned DPH result semantics."""
import json
import math
from pathlib import Path
import random
import re
import resource
import sqlite3
import sys
import time
from .vendor.dph_sql_executor import result_eq, replace_cur_year


def execute(payload):
    conn = sqlite3.connect(Path(payload['db_path']).resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    conn.execute('PRAGMA query_only=ON')
    conn.execute('PRAGMA temp_store=MEMORY')
    conn.text_factory = lambda b: b.decode(errors='ignore')
    if hasattr(conn,'setlimit'):
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 64*1024**2)
    def query(sql):
        deadline = time.monotonic() + float(payload['timeout'])
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        return conn.execute(replace_cur_year(sql)).fetchall()
    try:
        # A broken gold query is infrastructure failure, never model failure.
        try:
            gold = query(payload['gold'])
        except Exception as exc:
            return {'infrastructure_error': f'Spider gold: {type(exc).__name__}: {exc}'}
        try:
            prediction = query(payload['prediction'])
        except Exception as exc:
            return {'correct':False,'executable':False,'error':f'prediction:{type(exc).__name__}:{exc}'}
        ordered = 'orderby' in re.sub(r'\s+', '', payload['gold'].lower())
        # The upstream comparator samples constraints for >3 columns. Its
        # final equality is exact; isolate and seed its acceleration choices.
        random.seed(0)
        correct = bool(result_eq(prediction, gold, ordered))
        return {'correct':correct,'executable':True,'error':None,
                'predicted_rows':len(prediction),'gold_rows':len(gold),'order_matters':ordered}
    finally:
        conn.close()


def main():
    payload = json.loads(sys.stdin.read())
    memory = 2*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
    cpu = math.ceil(2*float(payload['timeout'])+5)
    resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    try:
        result = execute(payload)
    except Exception as exc:
        result = {'infrastructure_error':f'{type(exc).__name__}: {exc}'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
