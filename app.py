import os
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from collections import defaultdict, deque
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
import joblib
import tensorflow as tf

# バッファー初期化
buffers = defaultdict(lambda: deque(maxlen=100))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

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

def predict_sign(landmarks):
    """手話翻訳のメイン関数"""
    # 動きの判定ロジックをここに追加可能
    result = move_sign(landmarks)
    print(f"予測結果: {result}")
    return result

def move_sign(landmark_sequence):
    """手話の動きから分類を行う"""
    try:
        # NumPy配列に変換
        sequence_array = np.array(landmark_sequence, dtype=np.float32)
        
        # 入力形状の確認とログ出力
        print(f"変換前 shape: {sequence_array.shape}")
        
        # 正しい形状に変換 (1, 50, 63)
        if sequence_array.shape != (50, 63):
            # (50, 21, 3) → (50, 63) に変換
            sequence_array = sequence_array.reshape(50, -1)
        
        # バッチ次元を追加 (1, 50, 63)
        sequence_array = sequence_array.reshape(1, 50, 63)
        print(f"入力 shape: {sequence_array.shape}")
        
        # **修正点1: シンプルな予測実行（スコープ問題回避）**
        prediction = model(sequence_array, training=False)  # __call__メソッドを直接使用
        
        # NumPy配列に変換
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        
        # **修正点2: 予測結果から直接インデックスを取得**
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        print(f"🔍 予測確率: {prediction[0]}")
        print(f"🔍 予測ラベル: {predicted_label}")
        
        return predicted_label
        
    except Exception as e:
        print(f"⚠️ 予測エラー: {e}")
        return "予測エラー"

@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断時の処理"""
    sid = request.sid
    if sid in buffers:
        del buffers[sid]
        print(f"🔌 クライアント {sid} のバッファーを削除しました")

if __name__ == "__main__":
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
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)