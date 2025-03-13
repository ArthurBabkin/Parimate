class UserPhotoAdapter:
    def __init__(self, conn):
        self.conn = conn

    def insert_photo(self, user_id: int, embeddings):
        with self.conn:
            self.conn.execute(
                "INSERT INTO user_photos (id, ufile_id) VALUES (?, ?)",
                (user_id, embeddings))
