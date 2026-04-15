
import sqlite3
import pandas as pd

def connect():
    return sqlite3.connect("ml.db")

def create_table():
    conn = connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        input TEXT,
        prediction TEXT
    )""")
    conn.close()

def insert_prediction(data, pred):
    conn = connect()
    conn.execute("INSERT INTO predictions (input, prediction) VALUES (?, ?)", (str(data), str(pred)))
    conn.commit()
    conn.close()

def read_predictions():
    conn = connect()
    df = pd.read_sql("SELECT * FROM predictions", conn)
    conn.close()
    return df
def create_users_table():
    conn = connect()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT,
        password TEXT
    )
    """)
    conn.close()

def add_user(username, password):
    conn = connect()
    conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    data = cursor.fetchone()
    conn.close()
    return data