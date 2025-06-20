import google.generativeai as genai
import time
# ① Gemini APIキーをセット
genai.configure(api_key="AIzaSyDOKdOG61JLozonIWPBx1cZfGXaNvNWA78")
# ② モデルの準備（Gemini-Pro使用）
model = genai.GenerativeModel("models/gemini-2.5-pro")
def wordEdit(input_text):
    prompt = f"""
    以下の壊れた日本語の文字列を、意味が通じる自然な日本語に修正してください。
    文字の繰り返しや発音の揺れを含んでいても、文脈をもとに本来伝えたい日本語に変換してください。
    入力: {input_text}
    修正後の自然な日本語を1つだけ出力してください。
    """
    response = model.generate_content(prompt)
    return response.text.strip()
# ③ 成形したい文字列
print("成形したい文書を入力：")
input_text = input()
start = time.time()
print(f"修正後:{wordEdit(input_text)}")
end = time.time()
print(f"処理時間: {end - start:.5f} 秒")