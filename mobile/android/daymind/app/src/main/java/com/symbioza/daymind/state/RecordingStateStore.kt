package com.symbioza.daymind.state

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class RecordingStateStore {
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    fun markRecording() {
        _isRecording.value = true
    }

    fun markStopped() {
        _isRecording.value = false
    }
}
