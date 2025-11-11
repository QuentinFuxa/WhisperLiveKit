package com.symbioza.daymind.state

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.symbioza.daymind.DayMindApplication
import com.symbioza.daymind.audio.RecordingService
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class UiState(
    val isRecording: Boolean = false,
    val pendingChunks: Int = 0,
    val lastUploadMessage: String = "Waiting",
    val authError: Boolean = false
)

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val container = (getApplication() as DayMindApplication).container

    val uiState: StateFlow<UiState> = combine(
        container.recordingStateStore.isRecording,
        container.chunkRepository.pendingCount,
        container.uploadStatusStore.status
    ) { recording, pending, uploadStatus ->
        UiState(
            isRecording = recording,
            pendingChunks = pending,
            lastUploadMessage = uploadStatus.message,
            authError = uploadStatus.authError
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = UiState()
    )

    fun toggleRecording() {
        if (uiState.value.isRecording) {
            RecordingService.stop(getApplication())
        } else {
            RecordingService.start(getApplication())
        }
    }

    fun retryUploads() {
        container.uploadStatusStore.clearAuthError()
        viewModelScope.launch {
            container.chunkRepository.refresh()
            container.chunkUploadScheduler.enqueueAllPending()
        }
    }
}
