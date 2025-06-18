package com.example.ver1

import io.socket.client.IO
import io.socket.client.Socket
import org.json.JSONArray
import org.json.JSONObject

class LandmarkSocketIOClient(serverUrl: String, private val onResult: (String) -> Unit) {

    private var socket: Socket = IO.socket(serverUrl)

    fun connect() {
        socket.on(Socket.EVENT_CONNECT) {
            println("Socket.IO接続成功！")
        }.on(Socket.EVENT_DISCONNECT) {
            println("切断されました")
        }.on("result") { args ->
            if (args.isNotEmpty()) {
                val data = args[0] as JSONObject
                println("サーバーからの結果受信: $data")
                // 翻訳結果を取得してコールバック実行
                val text = data.optString("text", "")
                if (text.isNotEmpty()) {
                    onResult(text)
                }
            }
        }
        socket.connect()
    }

    fun sendLandmarks(landmarks: List<Landmark>) {
        val json = JSONObject()
        json.put("time", System.currentTimeMillis())

        val jsonArray = JSONArray()
        for (landmark in landmarks) {
            val lmJson = JSONObject()
            lmJson.put("x", landmark.x)
            lmJson.put("y", landmark.y)
            lmJson.put("z", landmark.z)
            jsonArray.put(lmJson)
        }
        json.put("landmarks", jsonArray)

        // landmarkイベントで送信
        socket.emit("landmark", json)
    }

    fun disconnect() {
        socket.disconnect()
    }
}

data class Landmark(val x: Float, val y: Float, val z: Float)
