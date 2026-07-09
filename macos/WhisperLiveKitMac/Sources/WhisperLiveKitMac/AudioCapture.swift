import AVFoundation
import Foundation

enum AudioCaptureError: LocalizedError {
    case permissionDenied
    case noInputDevice
    case unsupportedInputFormat
    case converterUnavailable

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Microphone permission was denied."
        case .noInputDevice:
            return "No microphone input device is available."
        case .unsupportedInputFormat:
            return "The current microphone format cannot be converted to 16 kHz mono PCM."
        case .converterUnavailable:
            return "The audio converter could not be created."
        }
    }
}

final class AudioCapture {
    private let engine = AVAudioEngine()
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: 16_000,
        channels: 1,
        interleaved: true
    )!

    private var converter: AVAudioConverter?
    private var isRunning = false

    func start(onPCMData: @escaping @Sendable (Data, Float) -> Void) async throws {
        guard await Self.requestMicrophoneAccess() else {
            throw AudioCaptureError.permissionDenied
        }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        guard inputFormat.channelCount > 0 else {
            throw AudioCaptureError.noInputDevice
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw AudioCaptureError.converterUnavailable
        }

        self.converter = converter
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { [weak self] buffer, _ in
            self?.process(buffer: buffer, inputFormat: inputFormat, onPCMData: onPCMData)
        }

        engine.prepare()
        try engine.start()
        isRunning = true
    }

    func stop() {
        guard isRunning else {
            return
        }

        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        isRunning = false
    }

    private func process(
        buffer: AVAudioPCMBuffer,
        inputFormat: AVAudioFormat,
        onPCMData: @escaping @Sendable (Data, Float) -> Void
    ) {
        guard let converter else {
            return
        }

        let frameRatio = targetFormat.sampleRate / max(inputFormat.sampleRate, 1)
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * frameRatio) + 512

        guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return
        }

        var didProvideInput = false
        var conversionError: NSError?

        let status = converter.convert(to: convertedBuffer, error: &conversionError) { _, outputStatus in
            if didProvideInput {
                outputStatus.pointee = .noDataNow
                return nil
            }

            didProvideInput = true
            outputStatus.pointee = .haveData
            return buffer
        }

        guard conversionError == nil else {
            return
        }

        switch status {
        case .haveData, .inputRanDry, .endOfStream:
            guard convertedBuffer.frameLength > 0, let data = Self.data(from: convertedBuffer) else {
                return
            }
            onPCMData(data, Self.rmsLevel(from: buffer))
        case .error:
            return
        @unknown default:
            return
        }
    }

    private static func data(from buffer: AVAudioPCMBuffer) -> Data? {
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers
        guard let bytes = audioBuffer.mData, audioBuffer.mDataByteSize > 0 else {
            return nil
        }
        return Data(bytes: bytes, count: Int(audioBuffer.mDataByteSize))
    }

    private static func rmsLevel(from buffer: AVAudioPCMBuffer) -> Float {
        guard let channels = buffer.floatChannelData else {
            return 0
        }

        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameCount > 0, channelCount > 0 else {
            return 0
        }

        var sum: Float = 0
        for channelIndex in 0..<channelCount {
            let samples = channels[channelIndex]
            for frameIndex in 0..<frameCount {
                let sample = samples[frameIndex]
                sum += sample * sample
            }
        }

        let mean = sum / Float(frameCount * channelCount)
        return min(max(sqrt(mean) * 5, 0), 1)
    }

    private static func requestMicrophoneAccess() async -> Bool {
        await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                continuation.resume(returning: granted)
            }
        }
    }
}
