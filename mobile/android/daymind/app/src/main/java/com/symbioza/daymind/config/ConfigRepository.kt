package com.symbioza.daymind.config

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.symbioza.daymind.BuildConfig

class ConfigRepository(context: Context) {
    private val encryptedPrefs = runCatching {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }.getOrNull()

    fun getServerUrl(): String = readOrDefault(KEY_SERVER_URL, BuildConfig.BASE_URL)

    fun getApiKey(): String = readOrDefault(KEY_API_KEY, BuildConfig.API_KEY)

    private fun readOrDefault(key: String, defaultValue: String): String {
        val value = encryptedPrefs?.getString(key, null)
        return value?.takeIf { it.isNotBlank() } ?: defaultValue
    }

    fun saveServerUrl(value: String) {
        encryptedPrefs?.edit()?.putString(KEY_SERVER_URL, value)?.apply()
    }

    fun saveApiKey(value: String) {
        encryptedPrefs?.edit()?.putString(KEY_API_KEY, value)?.apply()
    }

    companion object {
        private const val PREFS_NAME = "daymind.secrets"
        private const val KEY_SERVER_URL = "SERVER_URL"
        private const val KEY_API_KEY = "API_KEY"
    }
}
