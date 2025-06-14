from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from flask import request
from collections import defaultdict, deque

buffers = defaultdict(lambda: deque(maxlen=100))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # 任意のシークレットキー
socketio = SocketIO(app, cors_allowed_origins="*")  # CORSを許可（開発用）

@app.route("/")
def index():
    return render_template("index.html")

# クライアントから"landmark"イベントが送られたとき
@socketio.on("landmark")
def handle_landmark(data):
    print("受け取ったランドマーク:", data)
    sid = request.sid
    buffers[sid].append(data)

    if len(buffers[sid]) == 100:
        result = predict_sign(list(buffers[sid]))
        emit('result', result)
        buffers[sid].clear()

def predict_sign(landmarks):
    # ここに時系列モデルを使った処理を書く（仮のダミー）
    print("100フレームのランドマークを受け取りました。予測を実行します。")
    return {"text": "ありがとう"} 

#接続が切れたらbuffersを削除
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in buffers:
        del buffers[sid]


if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
