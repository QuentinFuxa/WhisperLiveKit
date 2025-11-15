package com.symbioza.daymind.data

import android.content.Context
import com.symbioza.daymind.audio.SpeechSegment
import java.io.File
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class ChunkRepository(context: Context) {
    private val chunksDir: File = File(context.cacheDir, "chunks").apply { mkdirs() }
    private val _pendingCount = MutableStateFlow(calculatePending())
    val pendingCount: StateFlow<Int> = _pendingCount.asStateFlow()
    private val _latestChunkPath = MutableStateFlow(findLatestChunkPath())
    val latestChunkPath: StateFlow<String?> = _latestChunkPath.asStateFlow()

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
        metadataFile(file).delete()
        refresh()
    }

    fun markChunkQueued() {
        refresh()
    }

    fun refresh() {
        _pendingCount.value = calculatePending()
        _latestChunkPath.value = findLatestChunkPath()
    }

    fun saveSpeechSegments(file: File, segments: List<SpeechSegment>) {
        val json = buildString {
            append("{\"segments\":[")
            segments.forEachIndexed { index, segment ->
                append("{\"start_ms\":${segment.startMs},\"end_ms\":${segment.endMs}}")
                if (index < segments.lastIndex) append(',')
            }
            append("]}")
        }
        metadataFile(file).writeText(json)
    }

    fun loadSpeechSegmentsJson(file: File): String? {
        val metadata = metadataFile(file)
        if (!metadata.exists()) return null
        return metadata.readText()
    }

    private fun calculatePending(): Int = listChunkFiles().size

    private fun findLatestChunkPath(): String? {
        return listChunkFiles().maxByOrNull { it.lastModified() }?.absolutePath
    }

    private fun metadataFile(file: File): File {
        return File(file.absolutePath + ".segments.json")
    }
}
