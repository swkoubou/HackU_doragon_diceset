package com.example.ver1

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.ver1.databinding.ActivitySignLanguageBinding
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream

class SignLanguageActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySignLanguageBinding
    private lateinit var handLandmarker: HandLandmarker
    private var isFrontCamera = true // 追加部分：カメラが前面か背面かを記録
    private lateinit var socketClient: LandmarkSocketIOClient
    private val serverUrl = "https://ryuichi.tail93a4a4.ts.net/"
    private var lastSentTime = 0L
    private val sendInterval = 500L // 500ms間隔で送信

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySignLanguageBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Socket.IOクライアントの初期化
        initializeSocketClient()

        // 追加部分：カメラ切り替えボタンの動作を設定
        binding.switchCameraButton.setOnClickListener {
            isFrontCamera = !isFrontCamera // フロント/バックを切り替え
            startCamera() // カメラ再起動
        }

        if (allPermissionsGranted()) {
            setupHandLandmarker()
            startCamera()
        } else {
            requestPermissions(REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun initializeSocketClient() {
        socketClient = LandmarkSocketIOClient(serverUrl) { translationResult ->
            // メインスレッドで UI を更新
            runOnUiThread {
                showTranslationResult(translationResult)
            }
        }
        socketClient.connect()
    }

    private fun setupHandLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result, _ ->
                runOnUiThread {
                    // ログ出力
                    result.landmarks().forEachIndexed { index, hand ->
                        Log.d("Hand $index", hand.joinToString {
                            "(${it.x()}, ${it.y()})"
                        })
                    }
                    // 追加部分：Overlay に結果を渡す
                    binding.overlay.setResults(result)

                    // ランドマークデータをサーバーに送信
                    sendLandmarksToServer(result)
                }
            }
            .build()

        handLandmarker = HandLandmarker.createFromOptions(this, options)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder().build().also {
                it.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
                    try {
                        val bitmap = imageProxyToBitmap(imageProxy, isFrontCamera) // 追加部分：isFrontCamera対応
                        val mpImage = BitmapImageBuilder(bitmap).build()
                        handLandmarker.detectAsync(mpImage, SystemClock.uptimeMillis())
                    } catch (e: Exception) {
                        Log.e("CameraAnalyzer", "Error processing image", e)
                    } finally {
                        imageProxy.close()
                    }
                }
            }

            val cameraSelector = if (isFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA // 追加部分：カメラ切り替え対応
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults) // 追加部分：super呼び出し
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) {
            setupHandLandmarker()
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    @OptIn(ExperimentalGetImage::class)
    fun imageProxyToBitmap(imageProxy: ImageProxy, isFrontCamera: Boolean): Bitmap { // 追加部分：isFrontCameraを受け取る
        val image = imageProxy.image ?: throw IllegalArgumentException("Image is null")
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()

        var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // 追加部分：回転補正とミラー反転処理
        val matrix = Matrix()
        matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        if (isFrontCamera) {
            matrix.postScale(-1f, 1f) // ミラー反転
        }
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        return bitmap
    }

    private fun sendLandmarksToServer(result: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastSentTime < sendInterval) return
        lastSentTime = currentTime

        if (result.landmarks().isEmpty()) return

        // 最初の手のランドマークを取得（21点）
        val handLandmarks = result.landmarks()[0]
        if (handLandmarks.size != 21) return

        // ローディング表示
        showLoading(true)

        val landmarks = handLandmarks.map { landmark ->
            Landmark(landmark.x(), landmark.y(), landmark.z())
        }

        try {
            socketClient.sendLandmarks(landmarks)
        } catch (e: Exception) {
            Log.e("SocketError", "Failed to send landmarks", e)
            showError("送信エラー: ${e.message}")
        }
    }

    private fun showLoading(show: Boolean) {
        binding.loadingIndicator.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun showTranslationResult(text: String) {
        binding.translationResult.text = text
        showLoading(false)

        // 結果をToastでも表示
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show()
    }

    private fun showError(message: String) {
        showLoading(false)
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::socketClient.isInitialized) {
            socketClient.disconnect()
        }
    }
}