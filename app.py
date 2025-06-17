from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from flask import request
from collections import defaultdict, deque
import handmark_1
import numpy as np

#
buffers = defaultdict(lambda: deque(maxlen=100))#データ格納庫　直近100フレーム
#各クライアントのランドマークを100フレーム分まで記録。
#自動で辞書の新しいキーに対して deque(maxlen=100) を初期化。
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # 任意のシークレットキー
socketio = SocketIO(app, cors_allowed_origins="*")  # CORSを許可（開発用）

@app.route("/")
def index():
    return render_template("index.html")#templates/index.htmlを表示

# クライアントから"landmark"イベントが送られたとき
@socketio.on("landmark")#クライアント（ブラウザやモバイルアプリ）から "landmark" イベントが送られると実行。
def handle_landmark(data):#data は30フレームの手のランドマーク情報
    #print("受け取ったランドマーク:", data)
    
    #リクエストしてきたユーザー
    sid = request.sid           #sid はWebSocketのクライアントごとの一意なセッションID
    buffers[sid].append(data)   #buffers[sid] にデータを蓄積（最大100フレームまで記録）
    #print(len(buffers[sid]))

    #30フレーム集まった時点で、predict_sign() に渡して手話の認識を行い、結果をクライアントに送信。
    if len(buffers[sid]) == 30:
        #手話翻訳関数からテキストを受け取る
        result = predict_sign(list(buffers[sid]))
        #モバイルへ送信
        emit('result', result)
        buffers[sid].clear()#追加した

#手話翻訳関数
def predict_sign(landmarks):
    #直近の30フレームから手が動いているか判別

    #動いてるなら

    #動いてないなら
    result = finger_classification(landmarks)#finger_classification() に渡して、何らかの特徴抽出 or 機械学習処理を予定。
    print("100フレームのランドマークを受け取りました。予測を実行します。")
    return {"text": result} 
def finger_classification(landmarks):
    # landmarksは30フレーム分のリスト。各要素は21点の(x, y, z)のリスト（例: [[(x1, y1, z1), ..., (x21, y21, z21)], ...]）

    # 最後のフレームを取得（最新の状態を使う）
    latest_frame = landmarks[-1]

    # tupleからオブジェクト化する（classify_handは landmarkオブジェクトのリストを想定しているので一時的な変換が必要）
    # → handmark_1.py側の `classify_hand()` を改修する場合はこの処理は不要にできる
    class Landmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    # x, y, z の座標から landmarkオブジェクトに変換
    landmark_objs = [Landmark(*point) for point in latest_frame]

    # 分類関数を呼び出し
    result = handmark_1.classify_hand(landmark_objs)

    return result

#接続が切れたらbuffersを削除
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in buffers:
        del buffers[sid]


if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
