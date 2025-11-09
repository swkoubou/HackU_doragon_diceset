import google.generativeai as genai
import pandas as pd
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

# Load configuration from local.properties
config = load_properties('server/local.properties')
API_KEY = config.get('API_KEY')

if not API_KEY:
    raise ValueError("API_KEY not found in server/local.properties. Please create the file and add 'API_KEY=your_key'.")

genai.configure(api_key=API_KEY)
# ② モデルの準備（Gemini-Pro使用）
model = genai.GenerativeModel("models/gemini-2.5-pro")
#model = genai.GenerativeModel("models/gemini-2.5-flash")
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
# ③ 成形したい文字列
print("成形したい文書を入力：")
input_text = input()
start = time.time()
print(f"修正後:{wordEdit(input_text)}")
end = time.time()
print(f"処理時間: {end - start:.5f} 秒")