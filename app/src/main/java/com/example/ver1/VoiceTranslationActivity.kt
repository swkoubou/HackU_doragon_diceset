package com.example.ver1

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.common.model.DownloadConditions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.tasks.await


class VoiceTranslationActivity : AppCompatActivity() {

    private lateinit var startRecognitionButton: Button
    private lateinit var stopRecognitionButton: Button
    private lateinit var resultText: TextView
    private lateinit var loadingProgressBar: ProgressBar
    private lateinit var backToMenuButton: Button
    private lateinit var retryButton: Button

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening = false

    // 権限リクエスト
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startSpeechRecognition()
        } else {
            Toast.makeText(this, "音声認識のためにマイクの権限が必要です", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_translation)

        initViews()
        setupClickListeners()
        setupSpeechRecognizer()
        resetScreen()
    }

    private fun initViews() {
        startRecognitionButton = findViewById(R.id.btn_record_voice)
        stopRecognitionButton = findViewById(R.id.btn_stop_record)
        resultText = findViewById(R.id.tv_voice_result)
        loadingProgressBar = findViewById(R.id.progress_voice_loading)
        backToMenuButton = findViewById(R.id.btn_voice_back_to_menu)
        retryButton = findViewById(R.id.btn_voice_record_again)

        stopRecognitionButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE
        retryButton.visibility = View.GONE
        loadingProgressBar.visibility = View.GONE
    }

    private fun setupClickListeners() {
        startRecognitionButton.setOnClickListener {
            checkPermissionAndStartRecognition()
        }

        stopRecognitionButton.setOnClickListener {
            stopSpeechRecognition()
        }

        backToMenuButton.setOnClickListener {
            finish()
        }

        retryButton.setOnClickListener {
            resetScreen()
        }
    }

    private fun setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Toast.makeText(this@VoiceTranslationActivity, "音声認識を開始します", Toast.LENGTH_SHORT).show()
                loadingProgressBar.visibility = View.VISIBLE
                resultText.text = "話してください..."
            }

            override fun onBeginningOfSpeech() {
                resultText.text = "認識中..."
            }

            override fun onRmsChanged(rmsdB: Float) {
                // 空実装
            }


            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                loadingProgressBar.visibility = View.GONE
            }

            override fun onError(error: Int) {
                loadingProgressBar.visibility = View.GONE
                isListening = false
                startRecognitionButton.visibility = View.VISIBLE
                stopRecognitionButton.visibility = View.GONE
                Toast.makeText(this@VoiceTranslationActivity, "認識エラーが発生しました: $error", Toast.LENGTH_SHORT).show()
                retryButton.visibility = View.VISIBLE
            }

            override fun onResults(results: Bundle?) {
                isListening = false
                loadingProgressBar.visibility = View.GONE
                startRecognitionButton.visibility = View.GONE
                stopRecognitionButton.visibility = View.GONE
                backToMenuButton.visibility = View.VISIBLE
                retryButton.visibility = View.VISIBLE

                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val recognizedText = matches[0]

                    // 「テスト」が認識された場合、反射神経ゲームを起動
                    if (recognizedText == "テスト") {
                        val intent = Intent(this@VoiceTranslationActivity, ReactionGameActivity::class.java)
                        startActivity(intent)
                        finish()
                        return
                    }

                    resultText.text = "認識結果:\n$recognizedText\n\n翻訳中..."
                    // 翻訳開始
                    translateText(recognizedText, "ja", "en")
                } else {
                    resultText.text = "音声を認識できませんでした。もう一度試してください。"
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun checkPermissionAndStartRecognition() {
        val audioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        if (audioPermission == PackageManager.PERMISSION_GRANTED) {
            startSpeechRecognition()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startSpeechRecognition() {
        if (isListening) return

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ja-JP") // 日本語固定
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            putExtra(RecognizerIntent.EXTRA_PROMPT, "話してください")
        }

        speechRecognizer?.startListening(intent)
        isListening = true

        startRecognitionButton.visibility = View.GONE
        stopRecognitionButton.visibility = View.VISIBLE
        backToMenuButton.visibility = View.GONE
        retryButton.visibility = View.GONE
        resultText.text = ""
    }

    private fun stopSpeechRecognition() {
        if (!isListening) return
        speechRecognizer?.stopListening()
        isListening = false
        startRecognitionButton.visibility = View.VISIBLE
        stopRecognitionButton.visibility = View.GONE
        backToMenuButton.visibility = View.VISIBLE
        retryButton.visibility = View.VISIBLE
        loadingProgressBar.visibility = View.GONE
    }

    private fun translateText(input: String, sourceLangCode: String, targetLangCode: String) {
        val options = TranslatorOptions.Builder()
            .setSourceLanguage(sourceLangCode)
            .setTargetLanguage(targetLangCode)
            .build()

        val translator = Translation.getClient(options)

        val conditions = DownloadConditions.Builder()
            .requireWifi()
            .build()

        CoroutineScope(Dispatchers.Main).launch {
            try {
                // モデルダウンロード（型明示: Unit）
                withContext<Unit>(Dispatchers.IO) {
                    translator.downloadModelIfNeeded(conditions).await()
                }

                // 翻訳（型明示: String）
                val translated: String = withContext(Dispatchers.IO) {
                    translator.translate(input).await()
                }

                resultText.text = "認識結果:\n$input\n\n翻訳結果:\n$translated"

            } catch (e: Exception) {
                resultText.text = "翻訳エラー: ${e.message}"
            }
        }
    }


    private fun resetScreen() {
        startRecognitionButton.visibility = View.VISIBLE
        stopRecognitionButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE
        retryButton.visibility = View.GONE
        loadingProgressBar.visibility = View.GONE
        resultText.text = "録音ボタンを押して音声認識を開始してください"
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer?.destroy()
    }

    override fun onPause() {
        super.onPause()
        if (isListening) {
            stopSpeechRecognition()
        }
    }
}
