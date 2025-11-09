from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from flask import request
from collections import defaultdict, deque
import random
import time
from datetime import datetime
import os

def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.
    """
    props = {}
    if not os.path.exists(filepath):
        print(f"Warning: Properties file '{filepath}' not found.")
        return props
    with open(filepath, "rt", encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep, 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip().strip('"')
                    props[key] = value
    return props

# ユーザーごとのデータ管理
user_data = defaultdict(lambda: {
    'frame_counter': 0,
    'landmarks_received': 0,
    'demo_words': ["こんにちは", "ありがとう", "おはよう", "さようなら", "はじめまして", "よろしく", "元気", "大丈夫"]
})

app = Flask(__name__)
config = load_properties('server/local.properties')
app.config['SECRET_KEY'] = config.get('SECRET_KEY', 'a-default-fallback-secret-key-for-development')
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "server on"

@socketio.on("landmark")
def handle_landmark(data):
    """ランドマークデータを受信してデモレスポンスを返す"""
    sid = request.sid
    user = user_data[sid]
    
    # フレームカウンター増加
    user['frame_counter'] += 1
    user['landmarks_received'] += 1
    
    # 受信確認ログ
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] ランドマーク受信 | ユーザー: {sid[:8]}... | フレーム: {user['frame_counter']}")
    
    # データの簡単な検証
    has_landmarks = data and 'landmarks' in data if isinstance(data, dict) else bool(data)
    
    # 10フレームごとにデモレスポンスを送信
    if user['frame_counter'] % 10 == 0:
        # ランダムな手話単語を選択
        demo_word = random.choice(user['demo_words'])
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        response_data = {
            'text': demo_word,
            'confidence': confidence,
            'frame_count': user['frame_counter'],
            'timestamp': current_time,
            'status': 'success' if has_landmarks else 'warning'
        }
        
        # クライアントにレスポンス送信
        emit('recognition_result', response_data)
        
        print(f"[{current_time}] 📤 送信: '{demo_word}' (信頼度: {confidence}) | 累積フレーム: {user['frame_counter']}")
    
    # 50フレームごとに文章デモ（文章終了のシミュレーション）
    if user['frame_counter'] % 50 == 0:
        # 複数の単語を組み合わせてデモ文章作成
        sentence_words = random.sample(user['demo_words'], 3)
        demo_sentence = " ".join(sentence_words)
        
        sentence_data = {
            'type': 'sentence_complete',
            'raw_text': demo_sentence,
            'refined_text': f"【AI整形】{demo_sentence}です。",
            'word_count': len(sentence_words),
            'total_frames': user['frame_counter'],
            'timestamp': current_time
        }
        
        emit('result', sentence_data)
        print(f"[{current_time}] 📝 文章完了: '{demo_sentence}' → '{sentence_data['refined_text']}'")

@socketio.on('connect')
def handle_connect():
    """クライアント接続時"""
    sid = request.sid
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] 🔗 新しいクライアント接続: {sid}")
    
    # 接続確認メッセージを送信
    emit('connection_status', {
        'status': 'connected',
        'message': 'サーバーに正常に接続されました',
        'session_id': sid,
        'timestamp': current_time
    })

@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断時"""
    sid = request.sid
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if sid in user_data:
        total_landmarks = user_data[sid]['landmarks_received']
        print(f"[{current_time}] 🔌 クライアント切断: {sid[:8]}... | 総受信フレーム: {total_landmarks}")
        del user_data[sid]
    else:
        print(f"[{current_time}] 🔌 クライアント切断: {sid[:8]}...")

@socketio.on('test_message')
def handle_test_message(data):
    """テスト用メッセージハンドラー"""
    sid = request.sid
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] 🧪 テストメッセージ受信: {data}")
    
    emit('test_response', {
        'message': 'テストメッセージを受信しました',
        'echo': data,
        'timestamp': current_time
    })

@socketio.on('reset_demo')
def handle_reset_demo():
    """デモリセット"""
    sid = request.sid
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if sid in user_data:
        user_data[sid]['frame_counter'] = 0
        user_data[sid]['landmarks_received'] = 0
        
    emit('reset_complete', {
        'message': 'デモがリセットされました',
        'timestamp': current_time
    })
    
    print(f"[{current_time}] 🔄 デモリセット: {sid[:8]}...")

# サーバー起動時のログ
@app.before_request
def setup_on_every_request():
    print("=" * 50)
    print("🚀 手話認識デモサーバー起動")
    print("📡 ポート: 8000")
    print("🌐 アクセス: http://localhost:8000")
    print("=" * 50)

if __name__ == "__main__":
    print("\n🎬 デモサーバー開始...")
    print("📱 モバイルアプリからランドマークを送信してください")
    print("⏱️  10フレームごとに認識結果をレスポンスします")
    print("📄 50フレームごとに文章完了をシミュレーションします")
    print("\n" + "="*50)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)