import os
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from collections import defaultdict, deque
#from tensorflow.keras.models import load_model
#from tensorflow.keras.initializers import Orthogonal
#import joblib
#import tensorflow as tf

# バッファー初期化
buffers = defaultdict(lambda: deque(maxlen=120))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

"""
# --- モデル読み込み ---
try:
    # tf.functionの問題を回避するため、compile=Falseを追加
    model = load_model('model/sign_model.h5', 
                      custom_objects={'Orthogonal': Orthogonal()},
                      compile=False)
    label_encoder = joblib.load('model/label_encoder.pkl')
    print("✅ モデルとラベルエンコーダーを読み込みました")
except Exception as e:
    print("❌ モデル読み込みエラー:", e)
    exit()
"""

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("landmark")
def handle_landmark(data):
    print("受け取ったランドマーク:", data)
    
    sid = request.sid
    buffers[sid].append(data)
    
    # 50フレームごと、10フレームずつスライドして予測
    buffer_len = len(buffers[sid])
    if buffer_len >= 50 and (buffer_len - 50) % 10 == 0:
        sequence = list(buffers[sid])[-50:]
        result = predict_sign(sequence)
        emit('result', result)

class MultiWindowRecognizer:
    def __init__(self):
        self.windows = [30, 50, 80]  # 120削除
        self.target_length = 55
        self.frame_buffer = deque(maxlen=80)  # 最大ウィンドウに合わせて調整
        self.model = load_model(input_shape=(55, 21, 3))
    
    def normalize_sequence_length(self, frames):
        frames = np.array(frames)
        
        if len(frames) > 55:
            # 長い場合：中央部分を抽出
            start = (len(frames) - 55) // 2
            return frames[start:start + 55]
        elif len(frames) < 55:
            # 短い場合：前後にゼロパディング
            pad_total = 55 - len(frames)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            
            padding_before = np.zeros((pad_before, 21, 3))
            padding_after = np.zeros((pad_after, 21, 3))
            
            return np.concatenate([padding_before, frames, padding_after])
        return frames
    
    def adjust_confidence(self, conf, window_size):
        # 55フレームに近いほど信頼度を上げる
        optimal_size = 55
        size_penalty = abs(window_size - optimal_size) / optimal_size * 0.15
        return max(0, conf - size_penalty)

@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断時の処理"""
    sid = request.sid
    if sid in buffers:
        del buffers[sid]
        print(f"🔌 クライアント {sid} のバッファーを削除しました")

if __name__ == "__main__":
    """
    # **修正点3: TensorFlowの設定を最適化**
    try:
        # Eager executionを有効化（デフォルトでONですが明示的に設定）
        tf.config.run_functions_eagerly(True)
        
        # GPUが利用可能な場合のメモリ制限設定
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU {gpu} のメモリ制限を設定しました")
    except Exception as e:
        print(f"⚠️ TensorFlow設定エラー: {e}")
    """
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5002)