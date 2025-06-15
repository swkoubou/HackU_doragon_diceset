from flask import Flask, request, render_template
import csv
import os
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    json_data = request.get_json()
    label = json_data["label"]
    data = json_data["data"]  # [フレーム][21点][x,y,z]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.csv"
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "point", "x", "y", "z", "label"])  # ヘッダー
        for frame_idx, frame in enumerate(data):
            for point_idx, (x, y, z) in enumerate(frame):
                writer.writerow([frame_idx, point_idx, x, y, z, label])

    return f"{filename} に保存しました。"

if __name__ == "__main__":
    app.run(debug=True, port=5050)
