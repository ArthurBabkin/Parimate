import sqlite3


def get_connection(db_path: str):
    conn = sqlite3.connect(db_path)

    return conn


def init_database(conn, path_db_init):
    with conn:
        with open(path_db_init, "r", encoding="utf-8") as file:
            conn.executescript(file.read())
