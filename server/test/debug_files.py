import os
import pickle
import joblib
import numpy as np

def check_file_info(file_path):
    """ファイルの基本情報をチェック"""
    if not os.path.exists(file_path):
        print(f"❌ ファイルが存在しません: {file_path}")
        return False
    
    size = os.path.getsize(file_path)
    print(f"📁 ファイル: {file_path}")
    print(f"📏 サイズ: {size} bytes")
    
    # ファイルの最初の数バイトを確認
    with open(file_path, 'rb') as f:
        first_bytes = f.read(10)
        print(f"🔍 最初の10バイト: {first_bytes}")
        print(f"🔍 16進数: {first_bytes.hex()}")
    
    return True

def try_load_pickle_methods(file_path):
    """複数の方法でpickleファイルの読み込みを試行"""
    methods = [
        ("standard pickle", lambda: pickle.load(open(file_path, 'rb'))),
        ("joblib", lambda: joblib.load(file_path)),
        ("numpy", lambda: np.load(file_path, allow_pickle=True)),
    ]
    
    for method_name, loader in methods:
        try:
            print(f"\n🔄 {method_name} で読み込み試行...")
            data = loader()
            print(f"✅ {method_name} で読み込み成功!")
            print(f"📊 データ型: {type(data)}")
            
            # データの内容を確認
            if hasattr(data, 'classes_'):
                print(f"📝 クラス数: {len(data.classes_)}")
                print(f"📝 クラス例: {data.classes_[:5]}")
            elif isinstance(data, (list, tuple, np.ndarray)):
                print(f"📝 データ長: {len(data)}")
                print(f"📝 データ例: {data[:5] if len(data) > 5 else data}")
            else:
                print(f"📝 データ: {data}")
                
            return data, method_name
            
        except Exception as e:
            print(f"❌ {method_name} で失敗: {e}")
    
    return None, None

def main():
    # ファイルパスを確認
    model_path = 'model/sign_model.h5'
    encoder_path = 'model/label_encoder.pkl'
    
    print("=" * 50)
    print("🔍 モデルファイルの検証")
    print("=" * 50)
    
    # モデルファイルの確認
    print("\n1. モデルファイルの確認")
    if check_file_info(model_path):
        print("✅ モデルファイルは存在します")
    
    # エンコーダーファイルの確認
    print("\n2. エンコーダーファイルの確認")
    if check_file_info(encoder_path):
        print("✅ エンコーダーファイルは存在します")
        
        # 複数の方法で読み込み試行
        data, successful_method = try_load_pickle_methods(encoder_path)
        
        if data is not None:
            print(f"\n✅ 読み込み成功: {successful_method}")
            print("\n📋 修正されたコード:")
            print_fixed_code(successful_method)
        else:
            print("\n❌ すべての方法で読み込み失敗")
            print("🔧 対処法:")
            print("1. ファイルが破損している可能性があります")
            print("2. ファイルを再生成してください")
            print("3. または、別の場所にバックアップがないか確認してください")

def print_fixed_code(method):
    """修正されたコードを表示"""
    if method == "joblib":
        print("""
# app.py の修正箇所:
import joblib  # 追加

# load_model_and_encoder関数内:
# 元のコード:
# with open(encoder_path, 'rb') as f:
#     label_encoder = pickle.load(f)

# 修正後:
label_encoder = joblib.load(encoder_path)
""")
    elif method == "numpy":
        print("""
# app.py の修正箇所:
import numpy as np  # 既に存在

# load_model_and_encoder関数内:
# 元のコード:
# with open(encoder_path, 'rb') as f:
#     label_encoder = pickle.load(f)

# 修正後:
label_encoder = np.load(encoder_path, allow_pickle=True)
""")

if __name__ == "__main__":
    main()