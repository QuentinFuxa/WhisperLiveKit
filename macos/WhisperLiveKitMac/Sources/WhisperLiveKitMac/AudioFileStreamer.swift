@preconcurrency import AVFoundation
import Foundation

enum AudioFileStreamerError: LocalizedError {
    case converterUnavailable
    case unreadableAudio

    var errorDescription: String? {
        switch self {
        case .converterUnavailable:
            return "The audio file cannot be converted to 16 kHz mono PCM."
        case .unreadableAudio:
            return "The selected audio file could not be read."
        }
    }
}

final class AudioFileStreamer {
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16_000,
        channels: 1,
        interleaved: true
    )!

    func streamRealtime(
        url: URL,
        chunkDuration: TimeInterval = 0.1,
        isPaused: @escaping () async -> Bool = { false },
        onPCMData: @escaping (Data, Float) -> Void
    ) async throws {
        let file = try AVAudioFile(forReading: url)
        let inputFormat = file.processingFormat

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw AudioFileStreamerError.converterUnavailable
        }

        let inputFramesPerChunk = AVAudioFrameCount(max(inputFormat.sampleRate * chunkDuration, 1_024))
        let silentChunk = Self.silentData(sampleRate: targetFormat.sampleRate, duration: chunkDuration)

        while file.framePosition < file.length && !Task.isCancelled {
            while await isPaused(), !Task.isCancelled {
                onPCMData(silentChunk, 0)
                try await Task.sleep(nanoseconds: UInt64(chunkDuration * 1_000_000_000))
            }

            let remainingFrames = AVAudioFrameCount(file.length - file.framePosition)
            let framesToRead = min(inputFramesPerChunk, remainingFrames)

            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: framesToRead) else {
                throw AudioFileStreamerError.unreadableAudio
            }

            try file.read(into: inputBuffer, frameCount: framesToRead)
            guard inputBuffer.frameLength > 0 else {
                break
            }

            let frameRatio = targetFormat.sampleRate / max(inputFormat.sampleRate, 1)
            let outputCapacity = AVAudioFrameCount(Double(inputBuffer.frameLength) * frameRatio) + 512

            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
                throw AudioFileStreamerError.unreadableAudio
            }

            var didProvideInput = false
            var conversionError: NSError?

            let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outputStatus in
                if didProvideInput {
                    outputStatus.pointee = .noDataNow
                    return nil
                }

                didProvideInput = true
                outputStatus.pointee = .haveData
                return inputBuffer
            }

            if let conversionError {
                throw conversionError
            }

            switch status {
            case .haveData, .inputRanDry, .endOfStream:
                if outputBuffer.frameLength > 0, let data = Self.data(from: outputBuffer) {
                    onPCMData(data, Self.rmsLevel(from: outputBuffer))
                    let duration = Double(outputBuffer.frameLength) / targetFormat.sampleRate
                    try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
                }
            case .error:
                throw AudioFileStreamerError.unreadableAudio
            @unknown default:
                break
            }
        }
    }

    private static func silentData(sampleRate: Double, duration: TimeInterval) -> Data {
        let sampleCount = max(Int(sampleRate * duration), 1)
        return Data(repeating: 0, count: sampleCount * MemoryLayout<Int16>.size)
    }

    private static func data(from buffer: AVAudioPCMBuffer) -> Data? {
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers
        guard let bytes = audioBuffer.mData, audioBuffer.mDataByteSize > 0 else {
            return nil
        }
        return Data(bytes: bytes, count: Int(audioBuffer.mDataByteSize))
    }

    private static func rmsLevel(from buffer: AVAudioPCMBuffer) -> Float {
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers
        guard let bytes = audioBuffer.mData, audioBuffer.mDataByteSize > 0 else {
            return 0
        }

        let sampleCount = Int(audioBuffer.mDataByteSize) / MemoryLayout<Int16>.size
        guard sampleCount > 0 else {
            return 0
        }

        let samples = bytes.bindMemory(to: Int16.self, capacity: sampleCount)
        var sum: Float = 0
        for index in 0..<sampleCount {
            let normalized = Float(samples[index]) / Float(Int16.max)
            sum += normalized * normalized
        }

        let mean = sum / Float(sampleCount)
        return min(max(sqrt(mean) * 5, 0), 1)
    }
}
