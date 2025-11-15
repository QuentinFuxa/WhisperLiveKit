package com.symbioza.daymind.audio

import android.content.Context
import android.media.MediaPlayer
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class ChunkPlaybackManager(private val context: Context) {
    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()

    private var mediaPlayer: MediaPlayer? = null

    fun play(file: File) {
        stop()
        if (!file.exists()) return
        val player = MediaPlayer()
        mediaPlayer = player
        player.setDataSource(file.absolutePath)
        player.setOnCompletionListener {
            stop()
        }
        player.setOnErrorListener { _, _, _ ->
            stop()
            true
        }
        player.prepare()
        player.start()
        _isPlaying.value = true
    }

    fun stop() {
        mediaPlayer?.run {
            runCatching { stop() }
            release()
        }
        mediaPlayer = null
        _isPlaying.value = false
    }
}
