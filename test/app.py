from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from collections import defaultdict, deque
import numpy as np
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
import os
import logging
import time
import google.generativeai as genai

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ユーザーごとのデータ管理
user_data = defaultdict(lambda: {
    'frame_buffer': deque(maxlen=100),
    'text_buffer': [],
    'error_counter': 0,
    'frame_counter': 0,
    'last_prediction_time': 0,
    'prediction_history': deque(maxlen=10),
    'static_landmark_counter': 0,  # 同じランドマークの連続カウント
    'last_landmark_data': None,    # 前回のランドマークデータ
    'static_threshold': 20         # 静的ランドマークの閾値
})

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

def load_model_and_encoder():
    try:
        model_path = 'model/sign_model.h5'
        encoder_path = 'model/label_encoder.pkl'

        if not os.path.exists(model_path):
            logger.error(f"モデルファイルが見つかりません: {model_path}")
            return None, None

        if not os.path.exists(encoder_path):
            logger.error(f"エンコーダーファイルが見つかりません: {encoder_path}")
            return None, None

        logger.info("モデルを読み込み中...")
        model = keras.models.load_model(model_path)
        logger.info(f"モデル読み込み完了 - 入力形状: {model.input_shape}")

        label_encoder = None
        try:
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("標準pickleで読み込み成功")
        except Exception as e1:
            logger.warning(f"標準pickle失敗: {e1}")
            try:
                label_encoder = joblib.load(encoder_path)
                logger.info("joblibで読み込み成功")
            except Exception as e2:
                logger.warning(f"joblib失敗: {e2}")
                return model, None

        if hasattr(label_encoder, 'classes_'):
            logger.info(f"クラス数: {len(label_encoder.classes_)}")
        return model, label_encoder

    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return None, None

def is_static_landmark(landmarks):
    """ランドマークが静的（すべて同じ値）かどうかを判定"""
    if not landmarks or len(landmarks) == 0:
        return False
    
    try:
        # すべてのランドマークが同じ値かチェック
        first_landmark = landmarks[0]
        for landmark in landmarks:
            if landmark != first_landmark:
                return False
        
        # さらに、[1, 0, 0]のような特定のパターンかチェック
        if first_landmark == [1, 0, 0]:
            return True
            
        return False
    except Exception as e:
        logger.error(f"静的ランドマーク判定エラー: {e}")
        return False

class MultiWindowRecognizer:
    def __init__(self):
        self.windows = [30, 50, 80]
        self.target_length = 55
        self.model, self.label_encoder = load_model_and_encoder()
        self.model_loaded = self.model is not None and self.label_encoder is not None

    def validate_landmark_data(self, frames):
        if not frames:
            return False, "フレームデータが空"
        try:
            frames_array = np.array(frames)
            if len(frames_array.shape) != 3 or frames_array.shape[1] != 21 or frames_array.shape[2] != 3:
                return False, "データ形状が不正"
            if np.isnan(frames_array).any():
                return False, "NaN値が含まれています"
            return True, "データ検証OK"
        except Exception as e:
            return False, f"検証エラー: {e}"

    def normalize_sequence_length(self, frames):
        frames = np.array(frames)
        if len(frames) > 55:
            start = (len(frames) - 55) // 2
            return frames[start:start + 55]
        elif len(frames) < 55:
            pad_total = 55 - len(frames)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            return np.concatenate([np.zeros((pad_before, 21, 3)), frames, np.zeros((pad_after, 21, 3))])
        return frames

    def adjust_confidence(self, conf, window_size):
        optimal_size = 55
        size_penalty = abs(window_size - optimal_size) / optimal_size * 0.15
        return max(0, conf - size_penalty)

    def select_best_result(self, results):
        if not results:
            return None, 0
        return max(results, key=lambda x: x[1])[:2]

    def predict(self, frame_buffer):
        if not self.model_loaded or len(frame_buffer) < min(self.windows):
            return None, 0
        frames_list = list(frame_buffer)
        valid, _ = self.validate_landmark_data(frames_list)
        if not valid:
            return None, 0
        results = []
        for window_size in self.windows:
            if len(frames_list) >= window_size:
                data = self.normalize_sequence_length(frames_list[-window_size:])
                try:
                    pred, conf = self.model_predict(data, window_size)
                    conf_adj = self.adjust_confidence(conf, window_size)
                    results.append((pred, conf_adj, window_size))
                except Exception as e:
                    logger.error(f"予測失敗: {e}")
        return self.select_best_result(results)

    def model_predict(self, data, window_size):
        try:
            input_data = data.reshape(1, data.shape[0], -1)
            predictions = self.model.predict(input_data, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            if hasattr(self.label_encoder, 'classes_') and class_idx < len(self.label_encoder.classes_):
                return self.label_encoder.classes_[class_idx], confidence
            return f"class_{class_idx}", confidence
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return self.mock_predict(data, window_size)

    def mock_predict(self, data, window_size):
        pred = np.random.choice(["こんにちは", "ありがとう", "さようなら"])
        conf = np.random.uniform(0.3, 0.9)
        logger.info(f"モック予測: {pred}, 信頼度: {conf:.3f}")
        return pred, conf

recognizer = MultiWindowRecognizer()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("landmark")
def handle_landmark(data):
    sid = request.sid
    user = user_data[sid]
    landmark_data = None
    print(f"Received data type: {type(data)}, data: {data}")
    print(f"Error counter: {user['error_counter']}, Static counter: {user['static_landmark_counter']}")

    if isinstance(data, list) and data:
        landmark_data = data
    elif isinstance(data, dict) and 'landmarks' in data:
        try:
            landmarks = data["landmarks"]
            landmark_data = landmarks
        except Exception as e:
            logger.error(f"ランドマーク変換エラー: {e}")
            user['error_counter'] += 1
            return
    else:
        user['error_counter'] += 1
        return

    if landmark_data is not None:
        # 静的ランドマークの検出
        if is_static_landmark(landmark_data):
            user['static_landmark_counter'] += 1
            logger.info(f"静的ランドマーク検出: {user['static_landmark_counter']}/{user['static_threshold']}")
            
            # 閾値に達した場合の処理
            if user['static_landmark_counter'] >= user['static_threshold']:
                logger.info("静的ランドマークが閾値に達しました。AIテキスト処理を開始します。")
                
                # テキストバッファにデータがある場合、AIで処理
                if user['text_buffer']:
                    raw_text = ' '.join(user['text_buffer'])
                    logger.info(f"AI処理開始 - Raw text: {raw_text}")
                    refined_text = refine_text_with_ai(raw_text)
                    
                    emit('result', {
                        'trigger': 'static_landmark_detected',
                        'raw_text': raw_text,
                        'refined_text': refined_text,
                        'prediction_history': list(user['prediction_history']),
                        'static_count': user['static_landmark_counter']
                    })
                    
                    # バッファをクリア
                    user['text_buffer'].clear()
                    user['prediction_history'].clear()
                else:
                    logger.info("テキストバッファが空のため、AI処理をスキップします。")
                    emit('result', {
                        'trigger': 'static_landmark_detected',
                        'message': '静的ランドマークを検出しましたが、処理するテキストがありません。',
                        'static_count': user['static_landmark_counter']
                    })
                
                # カウンターをリセット
                user['static_landmark_counter'] = 0
        else:
            # 静的でない場合はカウンターをリセット
            if user['static_landmark_counter'] > 0:
                logger.info("動的ランドマーク検出 - 静的カウンターをリセット")
                user['static_landmark_counter'] = 0
        
        # 通常の処理
        user['frame_buffer'].append(landmark_data)
        user['frame_counter'] += 1
        user['last_landmark_data'] = landmark_data
    else:
        user['error_counter'] += 1

    # 通常の予測処理
    if user['frame_counter'] % 5 == 0 and recognizer.model_loaded:
        now = time.time()
        if now - user['last_prediction_time'] > 0.5:
            user['last_prediction_time'] = now
            pred_text, conf = recognizer.predict(user['frame_buffer'])
            print(pred_text, conf)
            if pred_text and conf > 0.3:
                user['prediction_history'].append((pred_text, conf))
                user['text_buffer'].append(pred_text)
                emit('result', {
                    'text': pred_text,
                    'confidence': conf,
                    'frame_count': len(user['frame_buffer']),
                    'prediction_count': len(user['prediction_history'])
                })

    # 従来のエラーカウンターによる処理
    if user['error_counter'] >= 20:
        if user['text_buffer']:
            raw_text = ' '.join(user['text_buffer'])
            refined_text = refine_text_with_ai(raw_text)
            emit('result', {
                'trigger': 'error_threshold',
                'raw_text': raw_text,
                'refined_text': refined_text,
                'prediction_history': list(user['prediction_history'])
            })
            user['text_buffer'].clear()
            user['prediction_history'].clear()
        user['error_counter'] = 0

# Gemini APIキーをセット（環境変数や安全な方法で管理推奨）
genai.configure(api_key="AIzaSyDOKdOG61JLozonIWPBx1cZfGXaNvNWA78")

# Geminiモデルの準備（Gemini-Proなど）
model = genai.GenerativeModel("models/gemini-2.5-flash")

def wordEdit(input_text):
    prompt = f"""
    以下の壊れた日本語の文字列を、意味が通じる自然な日本語に修正してください。
    文字の繰り返しや発音の揺れを含んでいても、文脈をもとに本来伝えたい日本語に変換してください。
    以下間違いやすい文字です。
    あ、や
    う、ひ
    い、ち
    に、み
    入力: {input_text}
    修正後の自然な日本語を1つだけ出力してください。
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def refine_text_with_ai(raw_text):
    try:
        # Gemini APIを呼んで整形する
        print("AI開始")
        refined = wordEdit(raw_text)
        logger.info(f"AI整形結果: {refined}")
        return refined
    except Exception as e:
        logger.error(f"整形エラー: {e}")
        return raw_text

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    user_data.pop(sid, None)
    logger.info(f"切断: {sid}")

@socketio.on('reset_session')
def handle_reset():
    sid = request.sid
    if sid in user_data:
        user_data[sid] = {
            'frame_buffer': deque(maxlen=100),
            'text_buffer': [],
            'error_counter': 0,
            'frame_counter': 0,
            'last_prediction_time': 0,
            'prediction_history': deque(maxlen=10),
            'static_landmark_counter': 0,
            'last_landmark_data': None,
            'static_threshold': 20
        }
        emit('session_reset', {'status': 'success'})
        logger.info(f"セッションリセット: {sid}")

@socketio.on('get_debug_info')
def handle_debug_info():
    sid = request.sid
    user = user_data[sid]
    debug_info = {
        'model_loaded': recognizer.model_loaded,
        'frame_buffer_size': len(user['frame_buffer']),
        'text_buffer': user['text_buffer'],
        'error_counter': user['error_counter'],
        'frame_counter': user['frame_counter'],
        'prediction_history': list(user['prediction_history']),
        'static_landmark_counter': user['static_landmark_counter'],
        'static_threshold': user['static_threshold']
    }
    if recognizer.model_loaded:
        debug_info['model_input_shape'] = str(recognizer.model.input_shape)
        if hasattr(recognizer.label_encoder, 'classes_'):
            debug_info['class_count'] = len(recognizer.label_encoder.classes_)
    emit('debug_info', debug_info)

@socketio.on('set_static_threshold')
def handle_set_threshold(data):
    """静的ランドマークの閾値を動的に変更"""
    sid = request.sid
    user = user_data[sid]
    try:
        threshold = int(data.get('threshold', 20))
        user['static_threshold'] = max(1, min(threshold, 100))  # 1-100の範囲で制限
        emit('threshold_updated', {'threshold': user['static_threshold']})
        logger.info(f"静的ランドマーク閾値を更新: {user['static_threshold']}")
    except (ValueError, TypeError):
        emit('error', {'message': '無効な閾値です'})

if __name__ == "__main__":
    logger.info("アプリケーション開始")
    socketio.run(app, debug=True, host='0.0.0.0', port=4000)