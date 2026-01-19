import sqlite3
from pathlib import Path

def inspect(db_path):
    p=Path(db_path)
    print('\nDB:', p)
    if not p.exists():
        print('  MISSING')
        return
    conn=sqlite3.connect(str(p))
    cur=conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables=[t[0] for t in cur.fetchall()]
        print('  tables:', tables)
        cur.execute("SELECT source_type, COUNT(*) FROM data_provenance GROUP BY source_type")
        rows=cur.fetchall()
        if rows:
            for r in rows:
                print('  ', r[0], r[1])
        else:
            print('  no provenance rows')
    except Exception as e:
        print('  ERROR:', e)
    finally:
        conn.close()

inspect('data/subscription_fatigue_deployed.db')
inspect('data/subscription_fatigue.db')
