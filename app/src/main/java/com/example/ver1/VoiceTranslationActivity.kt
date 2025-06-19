package com.example.ver1

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale


class VoiceTranslationActivity : AppCompatActivity() {

    private lateinit var recordButton: Button
    private lateinit var stopButton: Button
    private lateinit var resultText: TextView
    private lateinit var loadingProgressBar: ProgressBar
    private lateinit var backToMenuButton: Button
    private lateinit var recordAgainButton: Button

    private var mediaRecorder: MediaRecorder? = null
    private var audioFile: File? = null
    private var isRecording = false
    private val client = OkHttpClient()
    private var speechRecognizer: SpeechRecognizer? = null


    // 権限リクエスト用のランチャー
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val audioPermissionGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false

        if (audioPermissionGranted) {
            startAudioRecording()
        } else {
            Toast.makeText(this, "音声録音の権限が必要です", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_voice_translation)

        initViews()
        setupClickListeners()

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Toast.makeText(this@VoiceTranslationActivity, "音声認識を開始します", Toast.LENGTH_SHORT).show()
            }

            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                Toast.makeText(this@VoiceTranslationActivity, "認識エラー: $error", Toast.LENGTH_SHORT).show()
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val recognizedText = matches[0]
                    showResult("音声認識結果:\n$recognizedText")
                } else {
                    showResult("音声を認識できませんでした")
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })


    }

    private fun initViews() {
        recordButton = findViewById(R.id.btn_record_voice)
        stopButton = findViewById(R.id.btn_stop_record)
        resultText = findViewById(R.id.tv_voice_result)
        loadingProgressBar = findViewById(R.id.progress_voice_loading)
        backToMenuButton = findViewById(R.id.btn_voice_back_to_menu)
        recordAgainButton = findViewById(R.id.btn_voice_record_again)

        // 初期状態の設定
        stopButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE
    }

    private fun setupClickListeners() {
        recordButton.setOnClickListener {
            checkPermissionsAndStart()
        }

        stopButton.setOnClickListener {
            stopAudioRecording()
        }

        backToMenuButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()
        }

        recordAgainButton.setOnClickListener {
            resetScreen()
            checkPermissionsAndStart()
        }
    }

    private fun checkPermissionsAndStart() {
        val audioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)

        when {
            audioPermission == PackageManager.PERMISSION_GRANTED -> {
                startAudioRecording()
            }
            else -> {
                requestPermissionLauncher.launch(arrayOf(
                    Manifest.permission.RECORD_AUDIO
                ))
            }
        }
    }

    private fun startAudioRecording() {
        try {
            // 録音ファイルの準備
            audioFile = File(externalCacheDir, "voice_recording.3gp")

            // MediaRecorderの設定
            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
                setOutputFile(audioFile?.absolutePath)

                prepare()
                start()
            }

            isRecording = true
            updateUIForRecording()

            Toast.makeText(this, "録音を開始しました", Toast.LENGTH_SHORT).show()

        } catch (e: IOException) {
            Toast.makeText(this, "録音の開始に失敗しました: ${e.message}", Toast.LENGTH_LONG).show()
            resetRecorder()
        } catch (e: Exception) {
            Toast.makeText(this, "エラーが発生しました: ${e.message}", Toast.LENGTH_LONG).show()
            resetRecorder()
        }
    }

    private fun stopAudioRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            isRecording = false

            Toast.makeText(this, "録音を停止しました", Toast.LENGTH_SHORT).show()

            // サーバーに音声ファイルを送信
            audioFile?.let { file ->
                uploadAudioToServer(file)
            }

        } catch (e: Exception) {
            Toast.makeText(this, "録音の停止でエラーが発生しました: ${e.message}", Toast.LENGTH_LONG).show()
            resetRecorder()
        }
    }

    private fun uploadAudioToServer(file: File) {
        showLoading("音声を解析中...")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                // マルチパートリクエストを作成
                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "audio",
                        file.name,
                        file.asRequestBody("audio/3gp".toMediaType())
                    )
                    .build()

                val request = Request.Builder()
                    .url("http://YOUR_FLASK_SERVER_URL/translate_voice") // 実際のサーバーURLに変更
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()

                withContext(Dispatchers.Main) {
                    hideLoading()
                    if (response.isSuccessful) {
                        val result = response.body?.string() ?: "結果を取得できませんでした"
                        showResult(result)
                    } else {
                        showError("サーバーエラー: ${response.code}")
                    }
                }

            } catch (e: IOException) {
                withContext(Dispatchers.Main) {
                    hideLoading()
                    showError("通信エラー: ${e.message}")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    hideLoading()
                    showError("エラーが発生しました: ${e.message}")
                }
            }
        }
    }

    private fun updateUIForRecording() {
        resultText.text = "録音中... 停止ボタンを押すか、自動で30秒後に停止します"
        recordButton.visibility = View.GONE
        stopButton.visibility = View.VISIBLE
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE

        // 30秒後に自動停止
        CoroutineScope(Dispatchers.Main).launch {
            delay(30000) // 30秒
            if (isRecording) {
                stopAudioRecording()
            }
        }
    }

    private fun showLoading(message: String) {
        loadingProgressBar.visibility = View.VISIBLE
        resultText.text = message
        recordButton.visibility = View.GONE
        stopButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE
    }

    private fun hideLoading() {
        loadingProgressBar.visibility = View.GONE
    }

    private fun showResult(result: String) {
        resultText.text = "翻訳結果:\n$result"
        recordButton.visibility = View.GONE
        stopButton.visibility = View.GONE
        backToMenuButton.visibility = View.VISIBLE
        recordAgainButton.visibility = View.VISIBLE
    }

    private fun showError(error: String) {
        resultText.text = "エラー: $error"
        recordButton.visibility = View.VISIBLE
        stopButton.visibility = View.GONE
        backToMenuButton.visibility = View.VISIBLE
        recordAgainButton.visibility = View.GONE
    }

    private fun resetScreen() {
        resultText.text = "録音ボタンを押して音声を録音してください"
        recordButton.visibility = View.VISIBLE
        stopButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE
        loadingProgressBar.visibility = View.GONE
    }

    private fun resetRecorder() {
        try {
            mediaRecorder?.release()
        } catch (e: Exception) {
            // 無視
        }
        mediaRecorder = null
        isRecording = false
        resetScreen()
    }

    private fun startSpeechRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, "話してください")
        }
        speechRecognizer?.startListening(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        resetRecorder()
        speechRecognizer?.destroy()
    }

    override fun onPause() {
        super.onPause()
        if (isRecording) {
            stopAudioRecording()
        }
    }
}