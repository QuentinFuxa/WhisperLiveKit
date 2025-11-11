package com.symbioza.daymind.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.UUID

class ChunkRepository(context: Context) {
    private val chunksDir: File = File(context.cacheDir, "chunks").apply { mkdirs() }
    private val _pendingCount = MutableStateFlow(calculatePending())
    val pendingCount: StateFlow<Int> = _pendingCount.asStateFlow()

    fun newChunkFile(): File {
        val chunkFile = File(chunksDir, "chunk_${System.currentTimeMillis()}_${UUID.randomUUID()}.wav")
        if (!chunkFile.exists()) {
            chunkFile.parentFile?.mkdirs()
            chunkFile.createNewFile()
        }
        return chunkFile
    }

    fun listChunkFiles(): List<File> {
        return chunksDir
            .listFiles { file -> file.extension.equals("wav", ignoreCase = true) }
            ?.sortedBy { it.name }
            ?: emptyList()
    }

    fun deleteChunk(file: File) {
        if (file.exists()) {
            file.delete()
        }
        refresh()
    }

    fun markChunkQueued() {
        refresh()
    }

    fun refresh() {
        _pendingCount.value = calculatePending()
    }

    private fun calculatePending(): Int = listChunkFiles().size
}
