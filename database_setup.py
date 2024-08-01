import sqlite3

conn=sqlite3.connect('users.db')
c=conn.cursor()

c.execute('''
    CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')
conn.commit()
conn.close()