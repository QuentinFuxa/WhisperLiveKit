package com.symbioza.daymind.upload

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import com.symbioza.daymind.config.ConfigRepository
import com.symbioza.daymind.data.ChunkRepository
import com.symbioza.daymind.device.DeviceIdProvider
import java.io.File
import java.time.Instant
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

class ChunkUploadScheduler(
    private val context: Context,
    private val chunkRepository: ChunkRepository,
    private val uploadStatusStore: UploadStatusStore,
    private val configRepository: ConfigRepository,
    private val deviceIdProvider: DeviceIdProvider
) {
    private val workManager = WorkManager.getInstance(context)

    fun scheduleChunkUpload(file: File, chunkStart: Instant) {
        if (uploadStatusStore.status.value.authError) {
            uploadStatusStore.markRetryableError("Upload paused due to auth error")
            chunkRepository.refresh()
            return
        }

        val baseUrl = configRepository.getServerUrl()
        val apiKey = configRepository.getApiKey()
        if (baseUrl.isBlank() || apiKey.isBlank()) {
            uploadStatusStore.markRetryableError("Missing BASE_URL/API_KEY config")
            chunkRepository.refresh()
            return
        }

        val speechSegmentsJson = chunkRepository.loadSpeechSegmentsJson(file)
        val dataBuilder = Data.Builder()
            .putString(ChunkUploadWorker.KEY_CHUNK_PATH, file.absolutePath)
            .putString(ChunkUploadWorker.KEY_SESSION_TS, DateTimeFormatter.ISO_INSTANT.format(chunkStart))
            .putString(ChunkUploadWorker.KEY_DEVICE_ID, deviceIdProvider.deviceId)
            .putInt(ChunkUploadWorker.KEY_SAMPLE_RATE, ChunkUploadWorker.DEFAULT_SAMPLE_RATE)
            .putString(ChunkUploadWorker.KEY_AUDIO_FORMAT, "wav")
        speechSegmentsJson?.let { dataBuilder.putString(ChunkUploadWorker.KEY_SPEECH_SEGMENTS, it) }
        val data = dataBuilder.build()

        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()

        val request = OneTimeWorkRequestBuilder<ChunkUploadWorker>()
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .setInputData(data)
            .build()

        workManager.enqueueUniqueWork(
            "upload-${file.name}",
            ExistingWorkPolicy.REPLACE,
            request
        )
        chunkRepository.markChunkQueued()
    }

    fun enqueueAllPending() {
        chunkRepository.listChunkFiles().forEach { file ->
            scheduleChunkUpload(file, Instant.now())
        }
    }
}
