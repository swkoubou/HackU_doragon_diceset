package com.example.ver1

import android.content.Context
import android.media.AudioAttributes
import android.media.SoundPool

class SoundManager private constructor(context: Context) {

    private var soundPool: SoundPool
    private var buttonClick50: Int = 0
    private var successSound: Int = 0
    private var buttonClick40: Int = 0
    private var excellentResult: Int = 0
    private var normalResult: Int = 0

    companion object {
        @Volatile
        private var INSTANCE: SoundManager? = null

        fun getInstance(context: Context): SoundManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: SoundManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    init {
        val audioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_GAME)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()

        soundPool = SoundPool.Builder()
            .setMaxStreams(3)
            .setAudioAttributes(audioAttributes)
            .build()

        // 効果音ファイルを読み込み
        buttonClick50 = soundPool.load(context, R.raw.button_click_40, 1)
        successSound = soundPool.load(context, R.raw.success, 1)
        buttonClick40 = soundPool.load(context, R.raw.button_click_50, 1)
        excellentResult = soundPool.load(context, R.raw.excellent_result, 1)
        normalResult = soundPool.load(context, R.raw.normal_result, 1)
    }

    fun playButtonClick50() {
        soundPool.play(buttonClick50, 1.0f, 1.0f, 1, 0, 1.0f)
    }

    fun playSuccessSound() {
        soundPool.play(successSound, 1.0f, 1.0f, 1, 0, 1.0f)
    }

    fun playButtonClick40() {
        soundPool.play(buttonClick40, 1.0f, 1.0f, 1, 0, 1.0f)
    }

    fun playResultSound(reactionTimeMs: Long) {
        val reactionTimeSeconds = reactionTimeMs / 1000.0
        if (reactionTimeSeconds <= 0.25) {
            soundPool.play(excellentResult, 1.0f, 1.0f, 1, 0, 1.0f)
        } else {
            soundPool.play(normalResult, 1.0f, 1.0f, 1, 0, 1.0f)
        }
    }

    fun release() {
        soundPool.release()
        INSTANCE = null
    }
}