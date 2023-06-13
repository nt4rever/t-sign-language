import sqlite3

# constant
IMAGE_SIZE = 64
MODEL_PATH = "../store/model/model_base.h5"
DB_PATH = "../store/database/gesture-30.db"


def load_labels(db_path="../store/database/gesture.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT g_id, g_name from gesture")
    labels = []
    for row in cursor:
        labels.append(row[1])
    return labels


# print(load_labels(DB_PATH))
