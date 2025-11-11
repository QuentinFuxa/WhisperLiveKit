package com.symbioza.daymind.upload

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class UploadStatus(
    val message: String = "",
    val lastError: String? = null,
    val authError: Boolean = false
)

class UploadStatusStore {
    private val _status = MutableStateFlow(UploadStatus(message = "Waiting"))
    val status: StateFlow<UploadStatus> = _status.asStateFlow()

    fun markSuccess(message: String) {
        _status.value = UploadStatus(message = message)
    }

    fun markRetryableError(message: String) {
        _status.value = UploadStatus(message = message, lastError = message)
    }

    fun markAuthError(message: String) {
        _status.value = UploadStatus(message = message, lastError = message, authError = true)
    }

    fun clearAuthError() {
        _status.value = UploadStatus(message = "Retrying uploads")
    }
}
