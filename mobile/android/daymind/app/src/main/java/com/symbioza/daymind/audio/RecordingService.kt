package com.symbioza.daymind.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.symbioza.daymind.AppContainer
import com.symbioza.daymind.DayMindApplication
import com.symbioza.daymind.MainActivity
import com.symbioza.daymind.R
import java.time.Instant
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RecordingService : Service() {
    private lateinit var container: AppContainer
    private val recordingFlag = AtomicBoolean(false)
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var recordingJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        container = (application as DayMindApplication).container
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startRecording()
            ACTION_STOP -> stopRecording()
        }
        return START_STICKY
    }

    override fun onDestroy() {
        stopRecording()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startRecording() {
        if (recordingFlag.get()) return
        recordingFlag.set(true)
        startForeground(NOTIFICATION_ID, buildNotification())
        container.recordingStateStore.markRecording()
        recordingJob = serviceScope.launch {
            recordContinuously()
        }
    }

    private fun stopRecording() {
        recordingFlag.set(false)
        recordingJob?.cancel()
        recordingJob = null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_DETACH)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        stopSelf()
        container.recordingStateStore.markStopped()
    }

    private suspend fun recordContinuously() = withContext(Dispatchers.IO) {
        val minBufferBytes = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBufferBytes == AudioRecord.ERROR || minBufferBytes == AudioRecord.ERROR_BAD_VALUE) {
            container.uploadStatusStore.markRetryableError("AudioRecord unavailable")
            stopRecording()
            return@withContext
        }

        val shortBufferSize = (minBufferBytes / 2).coerceAtLeast(1024)
        val buffer = ShortArray(shortBufferSize)
        val audioRecord = buildRecorder(minBufferBytes)
        var currentChunk = startChunk()
        try {
            audioRecord.startRecording()
            while (recordingFlag.get() && !Thread.currentThread().isInterrupted) {
                val read = audioRecord.read(buffer, 0, buffer.size)
                if (read > 0) {
                    currentChunk.writer.write(buffer, read)
                    currentChunk.samplesWritten += read
                    if (currentChunk.samplesWritten >= SAMPLES_PER_CHUNK) {
                        finalizeChunk(currentChunk)
                        currentChunk = startChunk()
                    }
                }
            }
        } catch (e: SecurityException) {
            container.uploadStatusStore.markRetryableError("Mic permission denied")
        } finally {
            finalizeChunk(currentChunk, keepIfEmpty = false)
            runCatching { audioRecord.stop() }
            audioRecord.release()
        }
    }

    private fun buildRecorder(bufferSizeBytes: Int): AudioRecord {
        val audioFormat = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(SAMPLE_RATE)
            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
            .build()
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(audioFormat)
                .setBufferSizeInBytes(bufferSizeBytes)
                .build()
        } else {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSizeBytes
            )
        }
    }

    private fun startChunk(): ActiveChunk {
        val file = container.chunkRepository.newChunkFile()
        val writer = WavWriter(file = file, sampleRate = SAMPLE_RATE)
        return ActiveChunk(
            file = file,
            writer = writer,
            sessionStart = Instant.now()
        )
    }

    private fun finalizeChunk(chunk: ActiveChunk, keepIfEmpty: Boolean = true) {
        runCatching { chunk.writer.close() }
        if (chunk.samplesWritten == 0L && !keepIfEmpty) {
            chunk.file.delete()
            container.chunkRepository.refresh()
            return
        }
        container.chunkUploadScheduler.scheduleChunkUpload(chunk.file, chunk.sessionStart)
    }

    private fun buildNotification(): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.notification_channel_desc)
        }
        val manager = ContextCompat.getSystemService(this, NotificationManager::class.java)
        manager?.createNotificationChannel(channel)
    }

    private data class ActiveChunk(
        val file: java.io.File,
        val writer: WavWriter,
        val sessionStart: Instant,
        var samplesWritten: Long = 0
    )

    companion object {
        private const val CHANNEL_ID = "daymind_recording"
        private const val NOTIFICATION_ID = 42
        private const val SAMPLE_RATE = 16_000
        // API contracts prefer ≤10 s chunks; align with the 6 s Python client.
        private const val CHUNK_SECONDS = 6
        private const val SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_SECONDS

        const val ACTION_START = "com.symbioza.daymind.action.START"
        const val ACTION_STOP = "com.symbioza.daymind.action.STOP"

        fun start(context: Context) {
            val intent = Intent(context, RecordingService::class.java).apply { action = ACTION_START }
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, RecordingService::class.java).apply { action = ACTION_STOP }
            context.startService(intent)
        }
    }
}
