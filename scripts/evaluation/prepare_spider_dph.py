#!/usr/bin/env python3
"""Import the exact DPH-RL Spider-1.0 dev split and official dev databases.

No model inference or training. Original DPH prompts/targets remain unchanged.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
DPH_COMMIT='fea6990642b072a59a1ecc653da31f9b876bd775'
DPH_SHA='fd8e1cfebc327b5d3ccb5b31216e25747fac5cb535b416ff4a19954419f34b46'
ZIP_SHA='00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b'
URL='https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing'
OUT=ROOT/'data/evaluation/spider-dev-dph-v1'
DATABASES=ROOT/'external/spider-data/spider-1.0-dev/database'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_exact(p,data):
    p=Path(p)
    if p.exists():
        if p.read_bytes()!=data:raise ValueError(f'Refusing to overwrite different bytes: {p}')
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.partial')
    temp.write_bytes(data);temp.replace(p)


def json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n').encode()


def question(row):
    return row['prompt'][0]['content'].split('\nQuestion:\n',1)[1].split('\n\nInstructions:',1)[0]


def gold_check(pair):
    i,row=pair
    database=DATABASES/row['db_id']/f"{row['db_id']}.sqlite"
    start=time.monotonic()
    conn=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        conn.set_progress_handler(lambda:int(time.monotonic()-start>15),10000)
        cur=conn.execute(row['query']);count=0
        while batch:=cur.fetchmany(1000):count+=len(batch)
        return {'id':i,'db_id':row['db_id'],'status':'ok','result_rows':count}
    except Exception as error:
        return {'id':i,'db_id':row['db_id'],'status':'error','error':f'{type(error).__name__}: {error}'}
    finally:conn.close()


def main():
    global ROOT, OUT, DATABASES
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--archive',type=Path)
    p.add_argument('--dph-source',type=Path,help='Pinned downloaded DPH files instead of a Git checkout')
    a=p.parse_args()
    ROOT=a.root.resolve(); OUT=ROOT/'data/evaluation/spider-dev-dph-v1'; DATABASES=ROOT/'external/spider-data/spider-1.0-dev/database'
    a.archive=(a.archive or ROOT/'external/spider-data/downloads/spider_data.zip').resolve()
    if sha(a.archive)!=ZIP_SHA:raise ValueError('Official archive digest differs from audited release')
    repo=a.dph_source.resolve() if a.dph_source else ROOT/'external/DPH-RL'
    if a.dph_source is None:
        commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if commit!=DPH_COMMIT:raise ValueError('Unexpected DPH checkout')
    dph=repo/'data/sql/spider_dev.jsonl'
    if sha(dph)!=DPH_SHA:raise ValueError('DPH JSONL differs from pinned release')
    rows=[json.loads(l) for l in dph.read_text().splitlines()]
    if [r['id'] for r in rows]!=list(range(1034)):raise ValueError('Unexpected DPH IDs/count')
    with zipfile.ZipFile(a.archive) as z:
        dev=json.loads(z.read('spider_data/dev.json'))
        if len(dev)!=len(rows):raise ValueError('Split count mismatch')
        for i,(r,o) in enumerate(zip(rows,dev)):
            gt=r['reward_model']['ground_truth']
            if (question(r),gt['db_id'],gt['gold_sql'],gt['cmp_method'])!=(o['question'],f"{o['db_id']}/{o['db_id']}.sqlite",o['query'],'spider'):
                raise ValueError(f'Official/DPH row differs: {i}')
        ids=sorted({r['db_id'] for r in dev})
        if len(ids)!=20:raise ValueError('Expected 20 dev databases')
        for origin,target in [('dev.json','official_dev.json'),('tables.json','official_tables.json'),
                              ('dev_gold.sql','official_dev_gold.sql'),('README.txt','OFFICIAL_README.txt')]:
            write_exact(OUT/target,z.read('spider_data/'+origin))
        tables=json.loads(z.read('spider_data/tables.json'))
        write_exact(OUT/'tables_dev.json',json_bytes([r for r in tables if r['db_id'] in ids]))
        for db in ids:
            # Exact allowlisted SQLite members only; no untrusted archive paths.
            member=f'spider_data/database/{db}/{db}.sqlite'
            write_exact(DATABASES/db/f'{db}.sqlite',z.read(member))
    write_exact(OUT/'dph_spider_dev.jsonl',dph.read_bytes())
    write_exact(OUT/'DPH_DATA_README.md',(repo/'README.md').read_bytes())
    write_exact(OUT/'DPH_LICENSE',(repo/'LICENSE').read_bytes())
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet=OUT/'dph_spider_dev.parquet'
    if not parquet.exists():
        temp=parquet.with_suffix('.parquet.partial')
        pq.write_table(pa.Table.from_pylist(rows),temp,compression='zstd');temp.replace(parquet)
    if pq.read_table(parquet).to_pylist()!=rows:raise ValueError('Parquet conversion changed rows')
    index=[{'id':i,'problem_id':f'spider:dev:{i}','db_id':r['db_id'],'question':r['question'],
            'gold_sql':r['query'],'database_path':str((DATABASES/r['db_id']/f"{r['db_id']}.sqlite").relative_to(ROOT))}
            for i,r in enumerate(dev)]
    write_exact(OUT/'index.jsonl',('\n'.join(json.dumps(r,ensure_ascii=False) for r in index)+'\n').encode())
    db_records={}
    for db in ids:
        file=DATABASES/db/f'{db}.sqlite'
        with sqlite3.connect(file.as_uri()+'?mode=ro',uri=True) as conn:
            integrity=conn.execute('PRAGMA quick_check').fetchall()
        if integrity!=[('ok',)]:raise ValueError(f'Database integrity failure: {db}')
        db_records[db]={'path':str(file.relative_to(ROOT)),'sha256':sha(file),'bytes':file.stat().st_size,'quick_check':'ok'}
    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(gold_check,enumerate(dev)))
    audit={'ordered_question_matches':1034,'ordered_database_id_matches':1034,'ordered_gold_sql_matches':1034,
        'dph_ids_contiguous':True,'parquet_roundtrip_equal':True,'databases_quick_check_ok':20,
        'gold_queries_ok':sum(r['status']=='ok' for r in checks),'gold_query_failures':[r for r in checks if r['status']!='ok'],
        'gold_query_timeout_seconds':15,'sqlite_version':sqlite3.sqlite_version,
        'database_question_counts':dict(sorted(Counter(r['db_id'] for r in dev).items()))}
    write_exact(OUT/'validation.json',json_bytes(audit))
    files={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file() and f.name not in ('manifest.json','README.md')}
    manifest={'schema_version':1,'release':'spider-1.0-dev-dph-v1','split':'official_dev','questions':1034,'databases':20,
        'dph_repository':'https://github.com/seamoke/DPH-RL','dph_commit':DPH_COMMIT,'dph_source_file':'data/sql/spider_dev.jsonl',
        'dph_source_sha256':DPH_SHA,'official_site':'https://yale-lily.github.io/spider','official_archive_url':URL,
        'official_archive_path':str(a.archive.relative_to(ROOT)) if a.archive.is_relative_to(ROOT) else str(a.archive),'official_archive_sha256':ZIP_SHA,
        'official_archive_bytes':a.archive.stat().st_size,'official_archive_annotation_notes':'README records 2020-08-03 schema and 2020-06-01 dev corrections; pinned archive first audited 2026-09-29.',
        'license':'Spider dataset: CC BY-SA 4.0; DPH repository: Apache-2.0. See original attribution files.',
        'files_sha256':files,'database_files':db_records,'validation':audit,
        'artifact_role':'Imported source data and original DPH prompts; dataset download only, no inference or job submission.',
        'format_note':'DPH README lists spider_dev.parquet; the pinned repository ships spider_dev.jsonl. Our parquet is a lossless local conversion.',
        'protocol_note':'DPH prompts request reasoning and fenced SQL. Preserve them as source; do not silently identify them with EvoRL direct raw-SQL prompts. DPH cmp_method=spider uses order-aware result matching, distinct from BIRD set equality and from multi-database test-suite accuracy.'}
    write_exact(OUT/'manifest.json',json_bytes(manifest))
    print(json.dumps({'output':str(OUT),'questions':len(rows),'databases':len(ids),'validation':audit},indent=2))


if __name__=='__main__':main()
