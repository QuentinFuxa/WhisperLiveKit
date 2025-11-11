package com.symbioza.daymind.device

import android.content.Context
import java.util.UUID

class DeviceIdProvider(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    val deviceId: String by lazy {
        prefs.getString(KEY_DEVICE_ID, null)?.takeIf { it.isNotBlank() }
            ?: UUID.randomUUID().toString().also { newId ->
                prefs.edit().putString(KEY_DEVICE_ID, newId).apply()
            }
    }

    companion object {
        private const val PREFS_NAME = "daymind_device"
        private const val KEY_DEVICE_ID = "device_id"
    }
}
