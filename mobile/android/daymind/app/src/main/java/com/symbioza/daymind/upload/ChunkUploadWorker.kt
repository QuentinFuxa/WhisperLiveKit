package com.symbioza.daymind.upload

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.symbioza.daymind.DayMindApplication
import java.io.File
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.scalars.ScalarsConverterFactory

class ChunkUploadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as DayMindApplication
        val container = app.container
        val chunkPath = inputData.getString(KEY_CHUNK_PATH) ?: return Result.failure()
        val sessionTs = inputData.getString(KEY_SESSION_TS) ?: return Result.failure()
        val deviceId = inputData.getString(KEY_DEVICE_ID) ?: return Result.failure()
        val sampleRate = inputData.getInt(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)
        val format = inputData.getString(KEY_AUDIO_FORMAT) ?: "wav"
        val speechSegments = inputData.getString(KEY_SPEECH_SEGMENTS)

        val chunkFile = File(chunkPath)
        if (!chunkFile.exists()) {
            container.chunkRepository.refresh()
            return Result.success()
        }

        val baseUrl = container.configRepository.getServerUrl().ensureTrailingSlash()
        val apiKey = container.configRepository.getApiKey()
        if (baseUrl.isBlank() || apiKey.isBlank()) {
            container.uploadStatusStore.markRetryableError("Missing BASE_URL/API_KEY config")
            return Result.retry()
        }

        val api = createApi(baseUrl, apiKey)
        val filePart = MultipartBody.Part.createFormData(
            "file",
            chunkFile.name,
            chunkFile.asRequestBody("audio/wav".toMediaType())
        )
        val textMediaType = "text/plain".toMediaType()

        val speechBody = speechSegments?.toRequestBody("application/json".toMediaType())

        return runCatching {
            api.uploadChunk(
                file = filePart,
                sessionTs = sessionTs.toRequestBody(textMediaType),
                deviceId = deviceId.toRequestBody(textMediaType),
                sampleRate = sampleRate.toString().toRequestBody(textMediaType),
                format = format.toRequestBody(textMediaType),
                speechSegments = speechBody
            )
        }.fold(
            onSuccess = { response ->
                when {
                    response.isSuccessful -> {
                        container.chunkRepository.deleteChunk(chunkFile)
                        container.uploadStatusStore.markSuccess("Uploaded ${chunkFile.name}")
                        Result.success()
                    }
                    response.code() == 401 || response.code() == 403 -> {
                        container.uploadStatusStore.markAuthError("Auth failed (${response.code()})")
                        Result.failure()
                    }
                    response.code() in 500..599 -> {
                        container.uploadStatusStore.markRetryableError("Server error ${response.code()}")
                        Result.retry()
                    }
                    else -> {
                        container.uploadStatusStore.markRetryableError("Upload failed ${response.code()}")
                        Result.retry()
                    }
                }
            },
            onFailure = { throwable ->
                container.uploadStatusStore.markRetryableError(throwable.message ?: "Network error")
                Result.retry()
            }
        )
    }

    private fun createApi(baseUrl: String, apiKey: String): TranscriptionApi {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }
        val client = OkHttpClient.Builder()
            .addInterceptor { chain ->
                val request = chain.request().newBuilder()
                    .addHeader("X-API-Key", apiKey)
                    .build()
                chain.proceed(request)
            }
            .addInterceptor(logging)
            .retryOnConnectionFailure(true)
            .callTimeout(1, TimeUnit.MINUTES)
            .build()

        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(client)
            .addConverterFactory(ScalarsConverterFactory.create())
            .build()
            .create(TranscriptionApi::class.java)
    }

    private fun String.ensureTrailingSlash(): String = if (endsWith('/')) this else "$this/"

    companion object {
        const val KEY_CHUNK_PATH = "chunk_path"
        const val KEY_SESSION_TS = "session_ts"
        const val KEY_DEVICE_ID = "device_id"
        const val KEY_SAMPLE_RATE = "sample_rate"
        const val KEY_AUDIO_FORMAT = "audio_format"
        const val KEY_SPEECH_SEGMENTS = "speech_segments"
        const val DEFAULT_SAMPLE_RATE = 16_000
    }
}
