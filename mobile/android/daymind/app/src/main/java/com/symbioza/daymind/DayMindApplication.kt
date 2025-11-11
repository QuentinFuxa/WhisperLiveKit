package com.symbioza.daymind

import android.app.Application
import com.symbioza.daymind.config.ConfigRepository
import com.symbioza.daymind.data.ChunkRepository
import com.symbioza.daymind.device.DeviceIdProvider
import com.symbioza.daymind.state.RecordingStateStore
import com.symbioza.daymind.upload.ChunkUploadScheduler
import com.symbioza.daymind.upload.UploadStatusStore

class DayMindApplication : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        container.chunkUploadScheduler.enqueueAllPending()
    }
}

class AppContainer(private val application: Application) {
    val configRepository = ConfigRepository(application)
    val chunkRepository = ChunkRepository(application)
    val uploadStatusStore = UploadStatusStore()
    val deviceIdProvider = DeviceIdProvider(application)
    val recordingStateStore = RecordingStateStore()
    val chunkUploadScheduler = ChunkUploadScheduler(
        context = application,
        chunkRepository = chunkRepository,
        uploadStatusStore = uploadStatusStore,
        configRepository = configRepository,
        deviceIdProvider = deviceIdProvider
    )
}
