import cv2
import mediapipe as mp
import pandas as pd
import os

# 保存先CSVファイル
csv_file = "data/hand_landmarks.csv"

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# カメラのセットアップ
cap = cv2.VideoCapture(0)

# 手の座標データを収集
landmarks_data = []

# 対象となる手話の静的五十音（動きのある「の・も・り・を・ん」、濁音・拗音は除外）
labels = [
    'あ', 'い', 'う', 'え', 'お',
    'か', 'き', 'く', 'け', 'こ',
    'さ', 'し', 'す', 'せ', 'そ',
    'た', 'ち', 'つ', 'て', 'と',
    'な', 'に', 'ぬ', 'ね',
    'は', 'ひ', 'ふ', 'へ', 'ほ',
    'ま', 'み', 'む', 'め',
    'や', 'ゆ', 'よ',
    'ら', 'る', 'れ', 'ろ',
    'わ'
]

# データ収集関数
def collect_landmarks(label):
    global landmarks_data
    count = 0
    print(f"\n=== Collecting data for '{label}' ===")
    print("Press 'q' to stop collecting...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # 画像保存用パスとフォルダ作成
        image_path = f'images/{label}/{count}.png'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, frame)

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks.append(label)
                landmarks_data.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            count += 1
            print(f"Collected {count} images for '{label}'")

        cv2.imshow('Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('Hand Landmarks', cv2.WND_PROP_VISIBLE) < 1:
            break

    print(f"\nFinished collecting {count} images for '{label}'.")
    cv2.destroyAllWindows()

# データ収集の実行
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

for label in labels:
    input(f"\nPress Enter to collect data for '{label}'...")
    os.makedirs(f'images/{label}', exist_ok=True)
    collect_landmarks(label)

# CSVファイルへの保存
try:
    columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
    df = pd.DataFrame(landmarks_data, columns=columns)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f'\nData saved successfully to {csv_file}')
except Exception as e:
    print(f'Error saving data: {e}')

# リソース解放
cap.release()
cv2.destroyAllWindows()
