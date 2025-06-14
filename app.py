from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from flask import request
from collections import defaultdict, deque

#
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

    #リクエストしてきたユーザー
    sid = request.sid
    buffers[sid].append(data)
    print(len(buffers[sid]))

    #buffersが30フレーム以上のデータを持つ
    if len(buffers[sid]) == 30:
        #手話翻訳関数からテキストを受け取る
        result = predict_sign(list(buffers[sid]))
        #モバイルへ送信
        emit('result', result)

#手話翻訳関数
def predict_sign(landmarks):
    #直近の30フレームから手が動いているか判別

    #動いてるなら

    #動いてないなら
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
