package com.example.ver1

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.random.Random

class ReactionGameActivity : AppCompatActivity() {

    private lateinit var soundManager: SoundManager
    private lateinit var titleText: TextView
    private lateinit var instructionText: TextView
    private lateinit var startButton: Button
    private lateinit var reactionButton: Button
    private lateinit var resultContainer: View
    private lateinit var resultText: TextView
    private lateinit var reactionTimeInResult: TextView
    private lateinit var closeResultButton: Button
    private lateinit var reactionTimeText: TextView
    private lateinit var bestTimeText: TextView
    private lateinit var averageTimeText: TextView
    private lateinit var playAgainButton: Button
    private lateinit var backToMenuButton: Button

    private var gameStartTime: Long = 0
    private var buttonShowTime: Long = 0
    private val handler = Handler(Looper.getMainLooper())
    private var waitingForReaction = false

    private lateinit var sharedPreferences: SharedPreferences

    companion object {
        private const val PREFS_NAME = "reaction_game_prefs"
        private const val KEY_BEST_TIME = "best_time"
        private const val KEY_TOTAL_TIME = "total_time"
        private const val KEY_GAME_COUNT = "game_count"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_reaction_game)

        // SoundManager初期化
        soundManager = SoundManager.getInstance(this)

        initViews()
        initSharedPreferences()
        setupClickListeners()
        showStartScreen()
    }

    private fun initViews() {
        titleText = findViewById(R.id.tv_game_title)
        instructionText = findViewById(R.id.tv_instruction)
        startButton = findViewById(R.id.btn_start_game)
        reactionButton = findViewById(R.id.btn_reaction)
        resultContainer = findViewById(R.id.result_container)
        resultText = findViewById(R.id.tv_result)
        reactionTimeInResult = findViewById(R.id.tv_reaction_time_in_result)
        closeResultButton = findViewById(R.id.btn_close_result)
        reactionTimeText = findViewById(R.id.tv_reaction_time)
        bestTimeText = findViewById(R.id.tv_best_time)
        averageTimeText = findViewById(R.id.tv_average_time)
        playAgainButton = findViewById(R.id.btn_play_again)
        backToMenuButton = findViewById(R.id.btn_back_to_menu)
    }

    private fun initSharedPreferences() {
        sharedPreferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
    }

    private fun setupClickListeners() {
        startButton.setOnClickListener {
            soundManager.playSuccessSound() // ゲーム開始ボタンは成功音
            startGame()
        }

        reactionButton.setOnClickListener {
            if (waitingForReaction) {
                soundManager.playButtonClick40() // ゲーム内ボタンは40
                handleReaction()
            }
        }

        playAgainButton.setOnClickListener {
            soundManager.playButtonClick40() // ゲーム内ボタンは40
            showStartScreen()
        }

        backToMenuButton.setOnClickListener {
            soundManager.playButtonClick40() // ゲーム内ボタンは40
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }

        closeResultButton.setOnClickListener {
            soundManager.playButtonClick40() // ×ボタンも40
            showStartScreen()
        }
    }

    private fun showStartScreen() {
        titleText.visibility = View.VISIBLE
        instructionText.visibility = View.VISIBLE
        startButton.visibility = View.VISIBLE
        reactionButton.visibility = View.GONE
        resultContainer.visibility = View.GONE
        reactionTimeText.visibility = View.GONE
        bestTimeText.visibility = View.GONE
        averageTimeText.visibility = View.GONE
        playAgainButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE

        titleText.text = "反射神経テスト"
        instructionText.text = "開始ボタンを押すと、1〜7秒後にボタンが表示されます。\nできるだけ早くボタンを押してください！"
        waitingForReaction = false
    }

    private fun startGame() {
        titleText.visibility = View.GONE
        instructionText.visibility = View.GONE
        startButton.visibility = View.GONE
        reactionButton.visibility = View.GONE
        resultText.visibility = View.VISIBLE
        resultText.text = "待機中..."

        gameStartTime = System.currentTimeMillis()
        val randomDelay = Random.nextLong(1000, 7001) // 1秒〜7秒

        handler.postDelayed({
            showReactionButton()
        }, randomDelay)
    }

    private fun showReactionButton() {
        resultText.text = "今だ！ボタンを押して！"
        reactionButton.visibility = View.VISIBLE
        reactionButton.text = "押せ！"
        buttonShowTime = System.currentTimeMillis()
        waitingForReaction = true
    }

    private fun handleReaction() {
        if (!waitingForReaction) return

        val reactionTime = System.currentTimeMillis() - buttonShowTime
        waitingForReaction = false

        saveResult(reactionTime)
        showResults(reactionTime)
    }

    private fun saveResult(reactionTime: Long) {
        val editor = sharedPreferences.edit()

        // ベストタイム更新
        val currentBest = sharedPreferences.getLong(KEY_BEST_TIME, Long.MAX_VALUE)
        if (reactionTime < currentBest) {
            editor.putLong(KEY_BEST_TIME, reactionTime)
        }

        // 平均計算用のデータ更新
        val totalTime = sharedPreferences.getLong(KEY_TOTAL_TIME, 0)
        val gameCount = sharedPreferences.getInt(KEY_GAME_COUNT, 0)

        editor.putLong(KEY_TOTAL_TIME, totalTime + reactionTime)
        editor.putInt(KEY_GAME_COUNT, gameCount + 1)

        editor.commit() // 同期処理で即座に保存
    }

    private fun showResults(reactionTime: Long) {
        Log.d("ReactionGame", "showResults called with time: $reactionTime")

        reactionButton.visibility = View.GONE

        // 他の要素を非表示
        titleText.visibility = View.GONE
        instructionText.visibility = View.GONE
        reactionTimeText.visibility = View.GONE
        bestTimeText.visibility = View.GONE
        averageTimeText.visibility = View.GONE
        playAgainButton.visibility = View.GONE
        backToMenuButton.visibility = View.GONE

        // 結果表示テキストも非表示
        resultText.visibility = View.GONE

        // 結果ウィンドウを表示
        resultContainer.visibility = View.VISIBLE
        Log.d("ReactionGame", "resultContainer visibility set to VISIBLE")

        // 反応時間表示（小数点以下3桁まで）
        val reactionTimeSeconds = reactionTime / 1000.0
        val timeText = "${String.format("%.3f", reactionTimeSeconds)}秒"
        reactionTimeInResult.text = timeText
        Log.d("ReactionGame", "Result text set to: $timeText")

        // 結果に応じた効果音を再生
        soundManager.playResultSound(reactionTime)
        Log.d("ReactionGame", "Playing result sound for time: $reactionTime ms")
    }

    override fun onBackPressed() {
        super.onBackPressed()
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }
}