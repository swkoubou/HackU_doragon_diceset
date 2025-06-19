from flask import Flask, render_template
from flask_socketio import SocketIO, emit, send
from flask import request
from collections import defaultdict, deque
import numpy as np
import pickle
import joblib
import tensorflow as tf
from tensorflow import keras
import os
import logging
import time

# ログレベルを詳細に設定
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
    'prediction_history': deque(maxlen=10)  # 最近の予測履歴
})

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

def load_model_and_encoder():
    """モデルとラベルエンコーダーの読み込み（改良版）"""
    try:
        model_path = 'model/sign_model.h5'
        encoder_path = 'model/label_encoder.pkl'
        
        if not os.path.exists(model_path):
            logger.error(f"モデルファイルが見つかりません: {model_path}")
            return None, None
            
        if not os.path.exists(encoder_path):
            logger.error(f"エンコーダーファイルが見つかりません: {encoder_path}")
            return None, None
        
        # モデルの読み込み
        logger.info("モデルを読み込み中...")
        model = keras.models.load_model(model_path)
        logger.info(f"モデル読み込み完了 - 入力形状: {model.input_shape}")
        
        # ラベルエンコーダーの読み込み
        logger.info("ラベルエンコーダーを読み込み中...")
        label_encoder = None
        
        # 方法1: 標準のpickle
        try:
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("標準pickleで読み込み成功")
        except Exception as e1:
            logger.warning(f"標準pickle失敗: {e1}")
            
            # 方法2: joblib
            try:
                label_encoder = joblib.load(encoder_path)
                logger.info("joblibで読み込み成功")
            except Exception as e2:
                logger.warning(f"joblib失敗: {e2}")
                return model, None
        
        if label_encoder is not None:
            if hasattr(label_encoder, 'classes_'):
                logger.info(f"クラス数: {len(label_encoder.classes_)}")
                logger.info(f"クラス一覧（最初の10個）: {label_encoder.classes_[:10]}")
                logger.info(f"クラス一覧（最後の10個）: {label_encoder.classes_[-10:]}")
            else:
                logger.info(f"エンコーダー型: {type(label_encoder)}")
        
        return model, label_encoder
        
    except Exception as e:
        logger.error(f"モデル読み込みエラー: {e}")
        return None, None

class MultiWindowRecognizer:
    def __init__(self):
        self.windows = [30, 50, 80]
        self.target_length = 55
        
        # モデルとエンコーダーの読み込み
        self.model, self.label_encoder = load_model_and_encoder()
        
        if self.model is None:
            logger.error("モデルの読み込みに失敗しました")
            self.model_loaded = False
        elif self.label_encoder is None:
            logger.error("エンコーダーの読み込みに失敗しました")
            self.model_loaded = False
        else:
            self.model_loaded = True
            logger.info("認識エンジンの初期化完了")
    
    def validate_landmark_data(self, frames):
        """ランドマークデータの検証"""
        if not frames:
            return False, "フレームデータが空"
        
        try:
            frames_array = np.array(frames)
            if len(frames_array.shape) != 3:
                return False, f"データ形状が不正: {frames_array.shape} (期待値: 3次元)"
            
            if frames_array.shape[1] != 21:  # 手のランドマークは21点
                return False, f"ランドマーク数が不正: {frames_array.shape[1]} (期待値: 21)"
            
            if frames_array.shape[2] != 3:  # x, y, z座標
                return False, f"座標数が不正: {frames_array.shape[2]} (期待値: 3)"
            
            # NaN値のチェック
            if np.isnan(frames_array).any():
                return False, "NaN値が含まれています"
            
            return True, "データ検証OK"
            
        except Exception as e:
            return False, f"データ検証エラー: {e}"
    
    def normalize_sequence_length(self, frames):
        """フレーム数を55に正規化"""
        frames = np.array(frames)
        
        if len(frames) > 55:
            start = (len(frames) - 55) // 2
            return frames[start:start + 55]
        elif len(frames) < 55:
            pad_total = 55 - len(frames)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            
            padding_before = np.zeros((pad_before, 21, 3))
            padding_after = np.zeros((pad_after, 21, 3))
            
            return np.concatenate([padding_before, frames, padding_after])
        return frames
    
    def adjust_confidence(self, conf, window_size):
        """ウィンドウサイズに応じて信頼度調整"""
        optimal_size = 55
        size_penalty = abs(window_size - optimal_size) / optimal_size * 0.15
        return max(0, conf - size_penalty)
    
    def select_best_result(self, results):
        """最適な結果を選択"""
        if not results:
            return None, 0
        
        # デバッグ情報を出力
        logger.info(f"予測結果候補: {[(r[0], f'{r[1]:.3f}', r[2]) for r in results]}")
        
        best = max(results, key=lambda x: x[1])
        return best[0], best[1]
    
    def predict(self, frame_buffer):
        """複数ウィンドウで予測実行"""
        if not self.model_loaded:
            logger.warning("モデルが読み込まれていません")
            return None, 0
            
        if len(frame_buffer) < min(self.windows):
            logger.debug(f"フレーム数不足: {len(frame_buffer)} < {min(self.windows)}")
            return None, 0
        
        # データ検証
        frames_list = list(frame_buffer)
        is_valid, message = self.validate_landmark_data(frames_list)
        if not is_valid:
            logger.warning(f"データ検証失敗: {message}")
            return None, 0
        
        results = []
        
        for window_size in self.windows:
            if len(frame_buffer) >= window_size:
                window_data = frames_list[-window_size:]
                normalized_data = self.normalize_sequence_length(window_data)
                
                # 実際のモデル予測
                try:
                    start_time = time.time()
                    pred, conf = self.model_predict(normalized_data, window_size)
                    prediction_time = time.time() - start_time
                    
                    adjusted_conf = self.adjust_confidence(conf, window_size)
                    results.append((pred, adjusted_conf, window_size))
                    
                    logger.debug(f"ウィンドウ{window_size}: {pred} (信頼度: {conf:.3f}→{adjusted_conf:.3f}, 時間: {prediction_time:.3f}s)")
                    
                except Exception as e:
                    logger.error(f"予測エラー (ウィンドウ{window_size}): {e}")
                    continue
        
        return self.select_best_result(results)
    
    def model_predict(self, normalized_data, window_size):
        """実際のモデル予測"""
        try:
            # データの形状を確認・調整
            logger.debug(f"正規化後データ形状: {normalized_data.shape}")
            
            if len(normalized_data.shape) == 3:
                # (frames, landmarks, coordinates) -> (1, frames, features)
                input_data = normalized_data.reshape(1, normalized_data.shape[0], -1)
            else:
                input_data = normalized_data.reshape(1, -1)
            
            logger.debug(f"モデル入力形状: {input_data.shape}")
            
            # モデルで予測
            predictions = self.model.predict(input_data, verbose=0)
            logger.debug(f"予測結果形状: {predictions.shape}")
            
            # 最も確率の高いクラスを取得
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # 上位3つの予測も表示（デバッグ用）
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_probs = predictions[0][top_3_indices]
            logger.debug(f"上位3予測: {[(idx, prob) for idx, prob in zip(top_3_indices, top_3_probs)]}")
            
            # ラベルエンコーダーでテキストに変換
            if hasattr(self.label_encoder, 'classes_'):
                if predicted_class_idx < len(self.label_encoder.classes_):
                    predicted_text = self.label_encoder.classes_[predicted_class_idx]
                else:
                    logger.error(f"クラスインデックス範囲外: {predicted_class_idx} >= {len(self.label_encoder.classes_)}")
                    predicted_text = f"unknown_class_{predicted_class_idx}"
            elif isinstance(self.label_encoder, (list, np.ndarray)):
                predicted_text = self.label_encoder[predicted_class_idx]
            else:
                predicted_text = f"class_{predicted_class_idx}"
            
            return predicted_text, confidence
            
        except Exception as e:
            logger.error(f"モデル予測エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # フォールバック：テスト用のダミー予測
            return self.mock_predict(normalized_data, window_size)
    
    def mock_predict(self, normalized_data, window_size):
        """モックの予測関数（エラー時のフォールバック）"""
        dummy_predictions = ["こんにちは", "ありがとう", "おはよう", "さようなら", "はじめまして"]
        import random
        pred = random.choice(dummy_predictions)
        conf = random.uniform(0.3, 0.9)
        logger.info(f"モック予測使用: {pred} (信頼度: {conf:.3f})")
        return pred, conf

# 認識エンジンのインスタンス
recognizer = MultiWindowRecognizer()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("landmark")
def handle_landmark(data):
    """ランドマークデータを受信"""
    sid = request.sid
    user = user_data[sid]
    
    # データの詳細ログ（最初の数回のみ）
    if user['frame_counter'] < 5:
        logger.info(f"ランドマーク受信 #{user['frame_counter']}: {sid}")
        if data:
            logger.info(f"データ型: {type(data)}, 長さ: {len(data) if hasattr(data, '__len__') else 'N/A'}")
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"最初の要素型: {type(data[0])}, サンプル: {data[0] if len(str(data[0])) < 100 else str(data[0])[:100] + '...'}")
    
    # データの検証を改良
    landmark_data = None
    if not data:
        user['error_counter'] += 1
        logger.debug(f"エラー: データが空 (エラー回数: {user['error_counter']})")
    elif isinstance(data, list) and len(data) > 0:
        # リスト形式のランドマークデータ
        landmark_data = data
    elif isinstance(data, dict) and 'landmarks' in data:
        # 辞書形式のランドマークデータ
        landmark_data = data['landmarks']
    else:
        user['error_counter'] += 1
        logger.debug(f"エラー: 無効なデータ形式 (エラー回数: {user['error_counter']})")
    
    if landmark_data is not None:
        user['frame_buffer'].append(landmark_data)
        user['error_counter'] = 0
    
    user['frame_counter'] += 1
    
    # 5フレームごとに認識実行（頻度を上げる）
    if user['frame_counter'] % 5 == 0 and recognizer.model_loaded:
        current_time = time.time()
        if current_time - user['last_prediction_time'] > 0.5:  # 0.5秒以上間隔を空ける
            user['last_prediction_time'] = current_time
            
            pred_text, confidence = recognizer.predict(user['frame_buffer'])
            
            # 信頼度の閾値を下げる（0.3以上）
            if pred_text and confidence > 0.3:
                user['prediction_history'].append((pred_text, confidence))
                user['text_buffer'].append(pred_text)
                
                logger.info(f"認識結果: {pred_text} (信頼度: {confidence:.3f})")
                
                emit('partial_result', {
                    'text': pred_text, 
                    'confidence': confidence,
                    'frame_count': len(user['frame_buffer']),
                    'prediction_count': len(user['prediction_history'])
                })
            else:
                logger.debug(f"信頼度不足: {pred_text} ({confidence:.3f})")
    
    # 文章終了判定
    if user['error_counter'] >= 20:  # しきい値を上げる
        if user['text_buffer']:
            raw_text = ' '.join(user['text_buffer'])
            refined_text = refine_text_with_ai(raw_text)
            
            emit('final_result', {
                'raw_text': raw_text,
                'refined_text': refined_text,
                'prediction_history': list(user['prediction_history'])
            })
            
            logger.info(f"文章完了: {raw_text} → {refined_text}")
            user['text_buffer'].clear()
            user['prediction_history'].clear()
        
        user['error_counter'] = 0

def refine_text_with_ai(raw_text):
    """生成AIで文章を整形"""
    try:
        # テスト用のダミー実装
        processed_text = raw_text.replace(' ', '')  # スペースを除去
        return f"【整形済み】{processed_text}"
    except Exception as e:
        logger.error(f"生成AI処理エラー: {e}")
        return raw_text

@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断時の処理"""
    sid = request.sid
    if sid in user_data:
        logger.info(f"クライアント切断: {sid}")
        del user_data[sid]

@socketio.on('reset_session')
def handle_reset():
    """セッションリセット"""
    sid = request.sid
    if sid in user_data:
        user_data[sid]['frame_buffer'].clear()
        user_data[sid]['text_buffer'].clear()
        user_data[sid]['error_counter'] = 0
        user_data[sid]['frame_counter'] = 0
        user_data[sid]['last_prediction_time'] = 0
        user_data[sid]['prediction_history'].clear()
        emit('session_reset', {'status': 'success'})
        logger.info(f"セッションリセット: {sid}")

@socketio.on('get_debug_info')
def handle_debug_info():
    """デバッグ情報を送信"""
    sid = request.sid
    user = user_data[sid]
    
    debug_info = {
        'model_loaded': recognizer.model_loaded,
        'frame_buffer_size': len(user['frame_buffer']),
        'text_buffer': user['text_buffer'],
        'error_counter': user['error_counter'],
        'frame_counter': user['frame_counter'],
        'prediction_history': list(user['prediction_history'])
    }
    
    if recognizer.model_loaded:
        debug_info['model_input_shape'] = str(recognizer.model.input_shape)
        if hasattr(recognizer.label_encoder, 'classes_'):
            debug_info['class_count'] = len(recognizer.label_encoder.classes_)
    
    emit('debug_info', debug_info)

if __name__ == "__main__":
    logger.info("アプリケーション開始")
    socketio.run(app, debug=True, host='0.0.0.0', port=4000)