// app/src/main/java/com/example/ver1/OverlayView.kt
package com.example.ver1

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var results: HandLandmarkerResult? = null
    private val paint = Paint().apply {
        color = Color.RED
        strokeWidth = 8f
        style = Paint.Style.FILL
    }

    fun setResults(handLandmarkerResult: HandLandmarkerResult) {
        results = handLandmarkerResult
        invalidate() // 再描画をトリガー
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        results?.let { result ->
            result.landmarks().forEach { hand ->
                hand.forEach { landmark ->
                    val x = landmark.x() * width
                    val y = landmark.y() * height
                    canvas.drawCircle(x, y, 10f, paint)
                }
            }
        }
    }
}