package com.symbioza.daymind.audio

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WavWriter(
    private val file: File,
    private val sampleRate: Int,
    private val channelCount: Int = 1,
    private val bitsPerSample: Int = 16
) {
    private val raf = RandomAccessFile(file, "rw")
    private var audioDataBytes: Long = 0

    init {
        writeHeaderPlaceholder()
    }

    val dataBytes: Long
        get() = audioDataBytes

    fun write(buffer: ShortArray, readCount: Int) {
        if (readCount <= 0) return
        val byteBuffer = ByteBuffer.allocate(readCount * 2)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until readCount) {
            byteBuffer.putShort(buffer[i])
        }
        raf.seek(raf.length())
        raf.write(byteBuffer.array())
        audioDataBytes += readCount * 2L
    }

    fun close() {
        updateHeader()
        raf.close()
    }

    private fun writeHeaderPlaceholder() {
        raf.setLength(0)
        raf.write(ByteArray(44))
    }

    private fun updateHeader() {
        val totalDataLen = audioDataBytes + 36
        val byteRate = sampleRate * channelCount * bitsPerSample / 8
        val header = ByteBuffer.allocate(44)
        header.order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(totalDataLen.toInt())
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1) // PCM
        header.putShort(channelCount.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort((channelCount * bitsPerSample / 8).toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(audioDataBytes.toInt())

        raf.seek(0)
        raf.write(header.array())
    }
}
