package com.example.ver1

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException

class SignLanguageActivity : AppCompatActivity() {

    private lateinit var recordButton: Button
    private lateinit var resultText: TextView
    private lateinit var loadingProgressBar: ProgressBar
    private lateinit var backToMenuButton: Button
    private lateinit var recordAgainButton: Button

    private var videoUri: Uri? = null
    private val client = OkHttpClient()

    // カメラ起動用のランチャー
    private val videoCapture = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            // 動画撮影完了後、サーバーに送信
            videoUri?.let { uri ->
                uploadVideoToServer(uri)
            }
        } else {
            hideLoading()
            Toast.makeText(this, "動画撮影がキャンセルされました", Toast.LENGTH_SHORT).show()
        }
    }

    // 権限リクエスト用のランチャー
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startVideoCapture()
        } else {
            Toast.makeText(this, "カメラの権限が必要です", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sign_language)

        initViews()
        setupClickListeners()
    }

    private fun initViews() {
        recordButton = findViewById(R.id.btn_record)
        resultText = findViewById(R.id.tv_result)
        loadingProgressBar = findViewById(R.id.progress_loading)
        backToMenuButton = findViewById(R.id.btn_back_to_menu)
        recordAgainButton = findViewById(R.id.btn_record_again)

        // 初期状態では結果関連のボタンを非表示
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE
    }

    private fun setupClickListeners() {
        recordButton.setOnClickListener {
            checkCameraPermissionAndStart()
        }

        backToMenuButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            // 既存のアクティビティスタックをクリアして新しくメイン画面を開く
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
            startActivity(intent)
            finish()
        }

        recordAgainButton.setOnClickListener {
            // 画面をリセットして再度録画可能にする
            resetScreen()
            checkCameraPermissionAndStart()
        }
    }

    private fun checkCameraPermissionAndStart() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                startVideoCapture()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun startVideoCapture() {
        showLoading("カメラを起動中...")

        val intent = Intent(MediaStore.ACTION_VIDEO_CAPTURE).apply {
            // 動画の保存先を指定
            val videoFile = File(externalCacheDir, "sign_language_video.mp4")
            videoUri = FileProvider.getUriForFile(
                this@SignLanguageActivity,
                "${packageName}.fileprovider",
                videoFile
            )
            putExtra(MediaStore.EXTRA_OUTPUT, videoUri)
            putExtra(MediaStore.EXTRA_DURATION_LIMIT, 30) // 最大30秒
        }

        if (intent.resolveActivity(packageManager) != null) {
            videoCapture.launch(intent)
        } else {
            hideLoading()
            Toast.makeText(this, "カメラアプリが見つかりません", Toast.LENGTH_SHORT).show()
        }
    }

    private fun uploadVideoToServer(uri: Uri) {
        showLoading("動画を解析中...")

        // コルーチンを使用して非同期処理
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val file = File(uri.path ?: "")

                // マルチパートリクエストを作成
                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "video",
                        file.name,
                        file.asRequestBody("video/mp4".toMediaType())
                    )
                    .build()

                val request = Request.Builder()
                    .url("http://YOUR_FLASK_SERVER_URL/translate") // 実際のサーバーURLに変更してください
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

    private fun showLoading(message: String) {
        loadingProgressBar.visibility = View.VISIBLE
        resultText.text = message
        recordButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE
    }

    private fun hideLoading() {
        loadingProgressBar.visibility = View.GONE
    }

    private fun showResult(result: String) {
        resultText.text = "翻訳結果:\n$result"
        recordButton.visibility = View.GONE
        backToMenuButton.visibility = View.VISIBLE
        recordAgainButton.visibility = View.VISIBLE
    }

    private fun showError(error: String) {
        resultText.text = "エラー: $error"
        recordButton.visibility = View.VISIBLE
        backToMenuButton.visibility = View.VISIBLE
        recordAgainButton.visibility = View.GONE
    }

    private fun resetScreen() {
        resultText.text = "録画ボタンを押して手話動画を撮影してください"
        recordButton.visibility = View.VISIBLE
        backToMenuButton.visibility = View.GONE
        recordAgainButton.visibility = View.GONE
        loadingProgressBar.visibility = View.GONE
    }
}