package com.example.ver1  // ← ver1 に合わせて変更

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

class HandOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var result: HandLandmarkerResult? = null

    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        strokeWidth = 10f
    }

    private val linePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    // ランドマーク21点をつなぐ接続情報
    private val connections = listOf(
        Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 4),
        Pair(0, 5), Pair(5, 6), Pair(6, 7), Pair(7, 8),
        Pair(5, 9), Pair(9, 10), Pair(10, 11), Pair(11, 12),
        Pair(9, 13), Pair(13, 14), Pair(14, 15), Pair(15, 16),
        Pair(13, 17), Pair(17, 18), Pair(18, 19), Pair(19, 20),
        Pair(0, 17)
    )

    fun setResults(result: HandLandmarkerResult?) {
        this.result = result
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        result?.let { res ->
            for (hand in res.landmarks()) {
                val points = hand.map {
                    PointF(it.x() * width, it.y() * height)
                }

                for (pt in points) {
                    canvas.drawCircle(pt.x, pt.y, 10f, paint)
                }

                for ((startIdx, endIdx) in connections) {
                    if (startIdx < points.size && endIdx < points.size) {
                        canvas.drawLine(
                            points[startIdx].x, points[startIdx].y,
                            points[endIdx].x, points[endIdx].y,
                            linePaint
                        )
                    }
                }
            }
        }
    }
}
