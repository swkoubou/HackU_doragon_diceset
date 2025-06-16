package com.example.ver1

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // UI要素を取得
        val signLanguageButton = findViewById<Button>(R.id.btn_sign_language)
        val voiceTranslationButton = findViewById<Button>(R.id.btn_voice_translation)




        // 手話翻訳ボタンのクリックリスナー
        signLanguageButton.setOnClickListener {
            val intent = Intent(this, SignLanguageActivity::class.java)
            startActivity(intent)
        }

        // 音声翻訳ボタンのクリックリスナー（今後実装予定）
        voiceTranslationButton.setOnClickListener {
            val intent = Intent(this,VoiceTranslationActivity::class.java)
            startActivity(intent)
        }
    }
}