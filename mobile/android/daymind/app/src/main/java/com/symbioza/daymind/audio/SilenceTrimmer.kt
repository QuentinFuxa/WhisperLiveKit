package com.symbioza.daymind.audio

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class SpeechSegment(val startMs: Long, val endMs: Long)

data class TrimResult(
    val keptSamples: Int,
    val segments: List<SpeechSegment>
)

object SilenceTrimmer {
    private const val HEADER_BYTES = 44
    private const val DEFAULT_THRESHOLD = 1200
    private const val DEFAULT_MIN_SPEECH_MS = 250L
    private const val DEFAULT_MIN_SILENCE_MS = 350L
    private const val DEFAULT_PADDING_MS = 150L

    fun trim(
        file: File,
        sampleRate: Int,
        threshold: Int = DEFAULT_THRESHOLD,
        minSpeechMs: Long = DEFAULT_MIN_SPEECH_MS,
        minSilenceMs: Long = DEFAULT_MIN_SILENCE_MS,
        paddingMs: Long = DEFAULT_PADDING_MS
    ): TrimResult {
        val bytes = file.readBytes()
        if (bytes.size <= HEADER_BYTES) {
            return TrimResult(0, emptyList())
        }
        val samples = extractSamples(bytes)
        val sampleSegments = detectSegments(samples, sampleRate, threshold, minSpeechMs, minSilenceMs)
        if (sampleSegments.isEmpty()) {
            return TrimResult(0, emptyList())
        }
        val trimmedSamples = collectTrimmedSamples(samples, sampleSegments, sampleRate, paddingMs)
        writeTrimmedFile(file, sampleRate, trimmedSamples)
        val speechSegments = sampleSegments.map { segment ->
            SpeechSegment(
                startMs = samplesToMs(segment.start, sampleRate),
                endMs = samplesToMs(segment.end, sampleRate)
            )
        }
        return TrimResult(trimmedSamples.size, speechSegments)
    }

    private fun extractSamples(bytes: ByteArray): ShortArray {
        val sampleCount = (bytes.size - HEADER_BYTES) / 2
        val buffer = ByteBuffer.wrap(bytes, HEADER_BYTES, bytes.size - HEADER_BYTES)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        val samples = ShortArray(sampleCount)
        for (i in 0 until sampleCount) {
            samples[i] = buffer.short
        }
        return samples
    }

    private data class SampleSegment(val start: Int, val end: Int)

    private fun detectSegments(
        samples: ShortArray,
        sampleRate: Int,
        threshold: Int,
        minSpeechMs: Long,
        minSilenceMs: Long
    ): List<SampleSegment> {
        val minSpeechSamples = (minSpeechMs * sampleRate / 1000).toInt().coerceAtLeast(1)
        val minSilenceSamples = (minSilenceMs * sampleRate / 1000).toInt().coerceAtLeast(1)
        val segments = mutableListOf<SampleSegment>()
        var isSpeaking = false
        var segmentStart = 0
        var lastSpeechSample = -1
        var silenceCounter = 0

        samples.forEachIndexed { index, sample ->
            val amplitude = abs(sample.toInt())
            if (amplitude >= threshold) {
                if (!isSpeaking) {
                    isSpeaking = true
                    segmentStart = index
                }
                lastSpeechSample = index
                silenceCounter = 0
            } else if (isSpeaking) {
                silenceCounter += 1
                if (silenceCounter >= minSilenceSamples) {
                    val endSample = max(segmentStart, lastSpeechSample)
                    if (endSample - segmentStart >= minSpeechSamples) {
                        segments.add(SampleSegment(segmentStart, endSample))
                    }
                    isSpeaking = false
                    silenceCounter = 0
                }
            }
        }

        if (isSpeaking) {
            val endSample = max(segmentStart, lastSpeechSample.takeIf { it >= 0 } ?: samples.lastIndex)
            if (endSample - segmentStart >= minSpeechSamples) {
                segments.add(SampleSegment(segmentStart, endSample))
            }
        }
        return segments
    }

    private fun collectTrimmedSamples(
        samples: ShortArray,
        segments: List<SampleSegment>,
        sampleRate: Int,
        paddingMs: Long
    ): ShortArray {
        val padSamples = (paddingMs * sampleRate / 1000).toInt().coerceAtLeast(0)
        val totalSamples = segments.sumOf { segment ->
            val start = max(0, segment.start - padSamples)
            val end = min(samples.size - 1, segment.end + padSamples)
            (end - start + 1).coerceAtLeast(0)
        }
        if (totalSamples <= 0) return ShortArray(0)

        val trimmed = ShortArray(totalSamples)
        var writeIndex = 0
        segments.forEach { segment ->
            val start = max(0, segment.start - padSamples)
            val end = min(samples.size - 1, segment.end + padSamples)
            for (i in start..end) {
                trimmed[writeIndex++] = samples[i]
            }
        }
        return trimmed
    }

    private fun writeTrimmedFile(file: File, sampleRate: Int, samples: ShortArray) {
        val writer = WavWriter(file, sampleRate)
        if (samples.isNotEmpty()) {
            writer.write(samples, samples.size)
        }
        writer.close()
    }

    private fun samplesToMs(sampleIndex: Int, sampleRate: Int): Long {
        return (sampleIndex * 1000L) / sampleRate
    }
}
