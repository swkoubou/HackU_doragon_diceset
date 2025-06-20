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
import math
import google.generativeai as genai

genai.configure(api_key="AIzaSyDOKdOG61JLozonIWPBx1cZfGXaNvNWA78")
model = genai.GenerativeModel("models/gemini-1.5-pro")

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
    'motion_history': deque(maxlen=20)  # 動的判定用
})

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

def wordEdit(input_text):
    prompt = f"""
以下の壊れた日本語の文字列を、意味が通じる自然な日本語に修正してください。
文字の繰り返しや発音の揺れを含んでいても、文脈をもとに本来伝えたい日本語に変換してください。
以下間違いやすい文字です：
- あ ↔ や
- う ↔ ひ
- い ↔ ち
- に ↔ み

入力: {input_text}
修正後の自然な日本語を1つだけ出力してください。
"""
    try:
        response = model.generate_content(prompt)
        # 最新のGemini APIではこう書くのが安全
        if hasattr(response, 'text') and response.text.strip():
            return response.text.strip()
        elif hasattr(response, 'parts'):
            return response.parts[0].text.strip()
        else:
            logger.warning("Gemini返答が空です")
            return input_text
    except Exception as e:
        logger.error(f"Gemini整形エラー: {e}")
        return input_text


class DynamicMotionDetector:
    """Stage 1: 動的判定（ルールベース）"""
    
    def __init__(self):
        self.motion_threshold = 0.02  # 動きの閾値
        self.stability_frames = 5     # 安定フレーム数
        
    def calculate_motion_variance(self, frames):
        """動きの分散を計算"""
        if len(frames) < 2:
            return 0
        
        frames_array = np.array(frames)
        # 連続フレーム間の差分を計算
        diffs = np.diff(frames_array, axis=0)
        # 各ランドマークの動きの大きさ
        motion_magnitudes = np.sqrt(np.sum(diffs**2, axis=2))
        # 全体の分散
        return np.var(motion_magnitudes)
    
    def calculate_trajectory_complexity(self, frames):
        """軌跡の複雑さを計算"""
        if len(frames) < 3:
            return 0
        
        frames_array = np.array(frames)
        # 主要な指先ランドマーク（人差し指、中指、薬指、小指の先端）
        key_points = [8, 12, 16, 20]  # MediaPipeの指先インデックス
        
        complexity_sum = 0
        for point_idx in key_points:
            if point_idx < frames_array.shape[1]:
                positions = frames_array[:, point_idx, :2]  # x, y座標のみ
                # 軌跡の曲率を計算
                if len(positions) >= 3:
                    curvature = self._calculate_curvature(positions)
                    complexity_sum += curvature
        
        return complexity_sum / len(key_points)
    
    def _calculate_curvature(self, positions):
        """軌跡の曲率を計算"""
        if len(positions) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(positions) - 1):
            p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
            
            # ベクトル計算
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 角度変化を計算
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle)
                curvatures.append(angle_change)
        
        return np.mean(curvatures) if curvatures else 0
    
    def is_dynamic_sign(self, frames):
        """動的手話かどうかを判定"""
        if len(frames) < 5:
            return False, 0.0, "フレーム数不足"
        
        # 特徴量計算
        motion_variance = self.calculate_motion_variance(frames)
        trajectory_complexity = self.calculate_trajectory_complexity(frames)
        
        # 判定ロジック
        is_dynamic = motion_variance > self.motion_threshold
        confidence = min(1.0, motion_variance / (self.motion_threshold * 2))
        
        reason = f"動き分散: {motion_variance:.4f}, 軌跡複雑さ: {trajectory_complexity:.4f}"
        
        return is_dynamic, confidence, reason

class HiraganaRecognizer:
    """Stage 2: ひらがな基本文字認識（既存のLSTMモデル）"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_loaded = False
        self.load_model()
        
    def load_model(self):
        """既存のLSTMモデルを読み込み"""
        try:
            model_path = 'model/sign_model.h5'
            encoder_path = 'model/label_encoder.pkl'
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                self.model = keras.models.load_model(model_path)
                
                try:
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                except:
                    self.label_encoder = joblib.load(encoder_path)
                
                self.model_loaded = True
                logger.info("ひらがな認識モデル読み込み完了")
            else:
                logger.warning("モデルファイルが見つかりません - デモモードで動作")
                
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
    
    def normalize_sequence(self, frames, target_length=55):
        """シーケンス長を正規化"""
        frames = np.array(frames)
        
        if len(frames) > target_length:
            start = (len(frames) - target_length) // 2
            return frames[start:start + target_length]
        elif len(frames) < target_length:
            pad_total = target_length - len(frames)
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            
            padding_before = np.zeros((pad_before, 21, 3))
            padding_after = np.zeros((pad_after, 21, 3))
            
            return np.concatenate([padding_before, frames, padding_after])
        return frames
    
    def predict(self, frames):
        try:
            normalized_data = self.normalize_sequence(frames)
            input_data = normalized_data.reshape(1, normalized_data.shape[0], -1)
            
            predictions = self.model.predict(input_data, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            if hasattr(self.label_encoder, 'classes_'):
                predicted_char = self.label_encoder.classes_[predicted_class_idx]
            else:
                predicted_char = f"class_{predicted_class_idx}"
            
            return predicted_char, confidence
            
        except Exception as e:
            logger.error(f"ひらがな予測エラー: {e}")
            return None, 0.0

class SpecialSymbolRecognizer:
    """Stage 3: 特殊記号認識（濁点、半濁点、小文字、長音）"""
    
    def __init__(self):
        # 現在はルールベースで実装、後でモデル化可能
        self.symbol_patterns = {
            'dakuten': {'motion_sharpness': 0.8, 'finger_curl': 0.6},     # 濁点
            'handakuten': {'motion_sharpness': 0.5, 'finger_curl': 0.3},  # 半濁点
            'small': {'hand_size_ratio': 0.7, 'motion_compact': 0.8},     # 小文字
            'chouon': {'horizontal_motion': 0.9, 'duration': 1.2}         # 長音
        }
    
    def analyze_finger_curl(self, frames):
        """指の曲がり具合を分析"""
        if len(frames) < 5:
            return 0.0
        
        frames_array = np.array(frames)
        # 指の関節間の角度を計算
        finger_tips = [4, 8, 12, 16, 20]  # 親指から小指の先端
        finger_bases = [2, 5, 9, 13, 17]  # 各指の付け根
        
        curl_scores = []
        for tip, base in zip(finger_tips, finger_bases):
            if tip < frames_array.shape[1] and base < frames_array.shape[1]:
                # 指先と付け根の距離変化
                distances = []
                for frame in frames_array:
                    dist = np.linalg.norm(frame[tip][:2] - frame[base][:2])
                    distances.append(dist)
                
                # 距離の変化量（曲がり具合の指標）
                curl_score = np.std(distances) / (np.mean(distances) + 1e-6)
                curl_scores.append(curl_score)
        
        return np.mean(curl_scores) if curl_scores else 0.0
    
    def analyze_motion_sharpness(self, frames):
        """動きの鋭さを分析"""
        if len(frames) < 3:
            return 0.0
        
        frames_array = np.array(frames)
        # 速度の急激な変化を検出
        velocities = np.diff(frames_array, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # 加速度の大きさ
        accel_magnitudes = np.sqrt(np.sum(accelerations**2, axis=(1, 2)))
        sharpness = np.max(accel_magnitudes) if len(accel_magnitudes) > 0 else 0.0
        
        return min(1.0, sharpness / 0.1)  # 正規化
    
    def analyze_hand_size(self, frames):
        """手のサイズ比率を分析"""
        if len(frames) < 1:
            return 1.0
        
        frames_array = np.array(frames)
        # 手の外接矩形を計算
        hand_sizes = []
        for frame in frames_array:
            hand_points = frame[:, :2]  # x, y座標のみ
            if len(hand_points) > 0:
                x_range = np.max(hand_points[:, 0]) - np.min(hand_points[:, 0])
                y_range = np.max(hand_points[:, 1]) - np.min(hand_points[:, 1])
                size = x_range * y_range
                hand_sizes.append(size)
        
        if not hand_sizes:
            return 1.0
        
        # 平均サイズに対する比率
        avg_size = np.mean(hand_sizes)
        return min(1.0, avg_size / 0.1)  # 正規化
    
    def detect_horizontal_motion(self, frames):
        """水平方向の動きを検出"""
        if len(frames) < 3:
            return 0.0
        
        frames_array = np.array(frames)
        # 重心の水平移動を計算
        centroids = np.mean(frames_array, axis=1)  # 各フレームの重心
        horizontal_movement = np.std(centroids[:, 0])  # x方向の標準偏差
        
        return min(1.0, horizontal_movement / 0.05)  # 正規化
    
    def predict(self, frames, base_char):
        """特殊記号を予測"""
        if not frames or not base_char:
            return None, 0.0
        
        # 各特徴量を計算
        finger_curl = self.analyze_finger_curl(frames)
        motion_sharpness = self.analyze_motion_sharpness(frames)
        hand_size_ratio = self.analyze_hand_size(frames)
        horizontal_motion = self.detect_horizontal_motion(frames)
        
        # 特殊記号の判定
        symbol_scores = {
            'dakuten': finger_curl * 0.6 + motion_sharpness * 0.4,
            'handakuten': finger_curl * 0.4 + motion_sharpness * 0.3 + (1 - motion_sharpness) * 0.3,
            'small': (1 - hand_size_ratio) * 0.7 + finger_curl * 0.3,
            'chouon': horizontal_motion * 0.8 + (1 - motion_sharpness) * 0.2,
            'none': 1.0 - max(finger_curl, motion_sharpness, 1-hand_size_ratio, horizontal_motion)
        }
        
        # 最高スコアの記号を選択
        best_symbol = max(symbol_scores.items(), key=lambda x: x[1])
        
        if best_symbol[0] == 'none' or best_symbol[1] < 0.4:
            return None, best_symbol[1]
        
        return best_symbol[0], best_symbol[1]

class ThreeStageRecognizer:
    """3段階統合認識システム"""
    
    def __init__(self):
        self.motion_detector = DynamicMotionDetector()
        self.hiragana_recognizer = HiraganaRecognizer()
        self.symbol_recognizer = SpecialSymbolRecognizer()
        
        # 濁点・半濁点変換テーブル
        self.dakuten_map = {
            'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
            'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
            'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
            'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ'
        }
        
        self.handakuten_map = {
            'は': 'ぱ', 'ひ': 'ぴ', 'ふ': 'ぷ', 'へ': 'ぺ', 'ほ': 'ぽ'
        }
        
        self.small_map = {
            'つ': 'っ', 'や': 'ゃ', 'ゆ': 'ゅ', 'よ': 'ょ'
        }
    
    def apply_symbol_transformation(self, base_char, symbol):
        """記号変換を適用"""
        if symbol == 'dakuten' and base_char in self.dakuten_map:
            return self.dakuten_map[base_char]
        elif symbol == 'handakuten' and base_char in self.handakuten_map:
            return self.handakuten_map[base_char]
        elif symbol == 'small' and base_char in self.small_map:
            return self.small_map[base_char]
        elif symbol == 'chouon':
            return base_char + 'ー'
        
        return base_char
    
    def predict(self, frame_buffer):
        if len(frame_buffer) < 10:
            return None, 0.0, "フレーム数不足"

        frames_list = list(frame_buffer)

        # Step 1: まず文字を認識
        base_char, hiragana_confidence = self.hiragana_recognizer.predict(frames_list)

        if not base_char or hiragana_confidence < 0.3:
            return None, hiragana_confidence, "ひらがな認識失敗"

        # Step 2: 動的かどうか判定（濁点などが必要か判別）
        is_dynamic, motion_confidence, motion_reason = self.motion_detector.is_dynamic_sign(frames_list)

        final_char = base_char
        total_confidence = hiragana_confidence  # 初期は基本文字のみの信頼度
        result_info = f"基本文字のみ: {base_char}"

        if is_dynamic:
            # Step 3: 動的なら記号を付加する
            symbol, symbol_confidence = self.symbol_recognizer.predict(frames_list, base_char)

            if symbol and symbol_confidence > 0.4:
                final_char = self.apply_symbol_transformation(base_char, symbol)
                total_confidence = hiragana_confidence * motion_confidence * symbol_confidence
                result_info = f"基本文字: {base_char}, 記号: {symbol}"
            else:
                total_confidence = hiragana_confidence * motion_confidence

        return final_char, total_confidence, result_info


# 認識エンジンのインスタンス
recognizer = ThreeStageRecognizer()

@app.route("/")
def index():
    return render_template('index.html')

@socketio.on("landmark")
def handle_landmark(data):
    """ランドマークデータを受信して3段階認識を実行"""
    sid = request.sid
    user = user_data[sid]
    
    # データ検証
    landmark_data_raw = None
    if isinstance(data, list) and len(data) > 0:
        landmark_data_raw = data
    elif isinstance(data, dict) and 'landmarks' in data:
        # Androidから送られてくる形式が {"landmarks": [[x,y,z], [x,y,z], ...]} の場合
        # または {"landmarks": [{"x": x, "y": y, "z": z}, ...]} の場合
        landmark_data_raw = data['landmarks']
    
    processed_landmark_frame = []
    if landmark_data_raw is not None:
        try:
            # ランドマークデータがリストのリスト（[[x,y,z],...]）か、
            # リストの辞書（[{'x':x,'y':y,'z':z},...]）かによって処理を分岐
            if all(isinstance(lm, list) for lm in landmark_data_raw):
                # 形式: [[x,y,z], [x,y,z], ...] の場合 (既に数値のリストなのでそのまま)
                processed_landmark_frame = [list(lm) for lm in landmark_data_raw]
            elif all(isinstance(lm, dict) and 'x' in lm and 'y' in lm and 'z' in lm for lm in landmark_data_raw):
                # 形式: [{'x': x, 'y': y, 'z': z}, ...] の場合
                # 各辞書からx, y, zの値を取り出してリストのリストに変換
                processed_landmark_frame = [[lm['x'], lm['y'], lm['z']] for lm in landmark_data_raw]
            else:
                logger.warning(f"未知のランドマークデータ形式: {landmark_data_raw}")
                user['error_counter'] += 1
                return # 処理を中断
                
            # 処理されたフレームをバッファに追加
            if processed_landmark_frame: # 空でないことを確認
                user['frame_buffer'].append(processed_landmark_frame)
                user['error_counter'] = 0
            else:
                # ランドマークが検出されなかった場合
                user['error_counter'] += 1
                logger.debug(f"ランドマークデータが空です (error_counter: {user['error_counter']})")

        except Exception as e:
            logger.error(f"ランドマークデータ処理エラー: {e}, データ: {landmark_data_raw}")
            user['error_counter'] += 1
            return # 処理を中断
    else:
        # データがNoneの場合（Androidからnullが送られてきた場合など）
        user['error_counter'] += 1
        logger.debug(f"受信したランドマークデータがNoneです (error_counter: {user['error_counter']})")
    
    user['frame_counter'] += 1
    
# app.py の handle_landmark 関数内 (一部抜粋)
    if user['frame_counter'] % 10 == 0:
        current_time = time.time()
        if current_time - user['last_prediction_time'] > 0.8:
            user['last_prediction_time'] = current_time
            
            logger.info(f"--- 認識処理開始 (SID: {sid}, フレーム数: {len(user['frame_buffer'])}) ---")
            pred_text, confidence, info = recognizer.predict(user['frame_buffer'])
            
            if pred_text and confidence > 0.2: # ここを一時的に低い値に
                user['prediction_history'].append((pred_text, confidence, info))
                user['text_buffer'].append(pred_text)
                
                logger.info(f"3段階認識結果: {pred_text} (信頼度: {confidence:.3f}) - {info}")
                
                emit('result', {
                    'text': pred_text,
                    'confidence': confidence,
                    'info': info,
                    'frame_count': len(user['frame_buffer']),
                    'stage': '3段階認識',
                    'timestamp': time.strftime('%H:%M:%S')
                })
            else:
                logger.info(f"認識失敗: {info} (信頼度: {confidence:.3f}), フレーム数: {len(user['frame_buffer'])}") # INFOレベルで詳細出力
    
    # 文章終了判定
    if user['error_counter'] >= 15:
        if user['text_buffer']:
            raw_text = ''.join(user['text_buffer'])  # 日本語なのでスペースなし
            refined_text = wordEdit(raw_text)
            
            emit('result', {
                'type': 'sentence_complete',
                'raw_text': raw_text,
                'refined_text': refined_text,
                'prediction_history': list(user['prediction_history'])[-5:],  # 最新5件
                'timestamp': time.strftime('%H:%M:%S')
            })
            
            logger.info(f"文章完了: {raw_text} → {refined_text}")
            user['text_buffer'].clear()
            user['prediction_history'].clear()
        
        user['error_counter'] = 0

@socketio.on('connect')
def handle_connect():
    """クライアント接続"""
    sid = request.sid
    logger.info(f"クライアント接続: {sid}")
    emit('connection_status', {
        'status': 'connected',
        'message': '3段階指文字認識システムに接続されました',
        'features': ['動的判定', 'ひらがな認識', '特殊記号認識'],
        'session_id': sid
    })

@socketio.on('disconnect')
def handle_disconnect():
    """クライアント切断"""
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
        emit('reset_complete', {'status': 'success'})
        logger.info(f"セッションリセット: {sid}")

if __name__ == "__main__":
    logger.info("3段階指文字認識サーバー開始")
    logger.info("Stage 1: 動的判定（ルールベース）")
    logger.info("Stage 2: ひらがな認識（既存LSTM）")
    logger.info("Stage 3: 特殊記号認識（ルールベース）")
    socketio.run(app, debug=True, host='0.0.0.0', port=4000)