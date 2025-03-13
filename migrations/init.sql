-- Enable the pgvector extension if not already enabled
-- CREATE EXTENSION IF NOT EXISTS vector;  -- Not valid in SQLite, comment out or remove

-- Create the 'users' table if it does not exist
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id INTEGER UNIQUE NOT NULL,
    embedding TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_photos (
    id INTEGER,
    ufile_id TEXT,
    embedding VECTOR
);

