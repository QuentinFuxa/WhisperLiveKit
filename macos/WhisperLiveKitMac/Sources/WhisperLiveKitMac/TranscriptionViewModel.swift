import AppKit
import Foundation
import UniformTypeIdentifiers

struct ConfigurationOption: Identifiable, Hashable {
    let id: String
    let title: String
}

@MainActor
final class TranscriptionViewModel: ObservableObject {
    @Published var endpoint = "ws://localhost:8000/asr"
    @Published var language = "server"
    @Published var modelFamily = "whisper" {
        didSet {
            normalizeConfiguration()
        }
    }
    @Published var modelSize = "base" {
        didSet {
            normalizeConfiguration()
        }
    }
    @Published var backend = "mlx-whisper" {
        didSet {
            normalizeConfiguration()
        }
    }
    @Published var streamingStrategy = "simulstreaming" {
        didSet {
            normalizeConfiguration()
        }
    }
    @Published var diarization = false {
        didSet {
            if !diarization {
                punctuationSplit = false
            }
        }
    }
    @Published var punctuationSplit = false
    @Published var phase: SessionPhase = .idle
    @Published var lines: [TranscriptLine] = []
    @Published var bufferTranscription = ""
    @Published var bufferDiarization = ""
    @Published var bufferTranslation = ""
    @Published var remainingTranscription = 0.0
    @Published var remainingDiarization = 0.0
    @Published var audioLevel: Float = 0
    @Published var elapsedSeconds: TimeInterval = 0
    @Published var statusText = "Ready"
    @Published var alert: AppAlert?
    @Published var selectedAudioFileName: String?

    let languages: [(id: String, title: String)] = [
        ("server", "Server default"),
        ("auto", "Auto detect"),
        ("en", "English"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("de", "German"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("nl", "Dutch"),
        ("ja", "Japanese"),
        ("zh", "Chinese")
    ]

    let modelFamilies = [
        ConfigurationOption(id: "whisper", title: "Whisper"),
        ConfigurationOption(id: "qwen3", title: "Qwen3-ASR"),
        ConfigurationOption(id: "voxtral", title: "Voxtral")
    ]

    private let transport = WebSocketTransport()
    private let audioCapture = AudioCapture()
    private let fileStreamer = AudioFileStreamer()
    private var startedAt: Date?
    private var timerTask: Task<Void, Never>?
    private var fileStreamTask: Task<Void, Never>?
    private var pendingInput: SessionInput = .microphone
    private var isNormalizingConfiguration = false

    var availableBackends: [ConfigurationOption] {
        switch modelFamily {
        case "qwen3":
            return [
                ConfigurationOption(id: "qwen3-vllm-metal", title: "vLLM Metal"),
                ConfigurationOption(id: "qwen3-vllm", title: "vLLM")
            ]
        case "voxtral":
            return [
                ConfigurationOption(id: "voxtral-mlx", title: "MLX"),
                ConfigurationOption(id: "voxtral", title: "Transformers")
            ]
        default:
            return [
                ConfigurationOption(id: "mlx-whisper", title: "MLX"),
                ConfigurationOption(id: "faster-whisper", title: "CTranslate2"),
                ConfigurationOption(id: "whisper", title: "PyTorch")
            ]
        }
    }

    var availableModelSizes: [ConfigurationOption] {
        switch modelFamily {
        case "qwen3":
            return [
                ConfigurationOption(id: "0.6b", title: "0.6B"),
                ConfigurationOption(id: "1.7b", title: "1.7B")
            ]
        case "voxtral":
            return []
        default:
            return [
                ConfigurationOption(id: "tiny", title: "tiny"),
                ConfigurationOption(id: "tiny.en", title: "tiny.en"),
                ConfigurationOption(id: "base", title: "base"),
                ConfigurationOption(id: "base.en", title: "base.en"),
                ConfigurationOption(id: "small", title: "small"),
                ConfigurationOption(id: "small.en", title: "small.en"),
                ConfigurationOption(id: "medium", title: "medium"),
                ConfigurationOption(id: "medium.en", title: "medium.en"),
                ConfigurationOption(id: "large-v3", title: "large-v3"),
                ConfigurationOption(id: "large-v3-turbo", title: "large-v3-turbo")
            ]
        }
    }

    var availableStreamingStrategies: [ConfigurationOption] {
        switch modelFamily {
        case "qwen3":
            return [
                ConfigurationOption(id: "earlycut", title: "Early cut")
            ]
        case "voxtral":
            return [
                ConfigurationOption(id: "native", title: "Native streaming")
            ]
        default:
            return [
                ConfigurationOption(id: "simulstreaming", title: "SimulStreaming"),
                ConfigurationOption(id: "localagreement", title: "LocalAgreement")
            ]
        }
    }

    var modelSizeControlTitle: String {
        switch modelFamily {
        case "voxtral":
            return "Voxtral Mini Realtime"
        default:
            return modelSize
        }
    }

    var canStart: Bool {
        switch phase {
        case .idle, .failed:
            return true
        case .connecting, .recording, .simulatingFile, .finalizing:
            return false
        }
    }

    var isInputActive: Bool {
        phase == .recording || phase == .simulatingFile
    }

    var localServerCommand: String {
        var parts = ["wlk"]

        parts.append("--backend \(backend)")
        if modelFamily != "voxtral" {
            parts.append("--model \(modelSize)")
        }
        if modelFamily == "whisper" {
            parts.append("--backend-policy \(streamingStrategy)")
        }
        parts.append("--language \(serverLanguage)")
        parts.append("--pcm-input")

        if diarization {
            parts.append("--diarization")
        }
        if punctuationSplit {
            parts.append("--punctuation-split")
        }

        return parts.joined(separator: " ")
    }

    func normalizeConfiguration() {
        guard !isNormalizingConfiguration else {
            return
        }

        isNormalizingConfiguration = true
        defer {
            isNormalizingConfiguration = false
        }

        let backends = availableBackends
        if !backends.contains(where: { $0.id == backend }), let first = backends.first {
            backend = first.id
        }

        let sizes = availableModelSizes
        if !sizes.isEmpty, !sizes.contains(where: { $0.id == modelSize }), let first = sizes.first {
            modelSize = first.id
        }

        let strategies = availableStreamingStrategies
        if !strategies.contains(where: { $0.id == streamingStrategy }), let first = strategies.first {
            streamingStrategy = first.id
        }
    }

    var transcriptText: String {
        var parts = lines.compactMap { line -> String? in
            guard line.speaker != -2, let text = line.text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
                return nil
            }

            let speaker = "Speaker \(line.speaker)"
            let time = [line.start, line.end].compactMap { $0 }.joined(separator: " - ")
            return time.isEmpty ? "\(speaker): \(text)" : "\(speaker) [\(time)]: \(text)"
        }

        let pending = [bufferDiarization, bufferTranscription]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")

        if !pending.isEmpty {
            parts.append("Live: \(pending)")
        }

        return parts.joined(separator: "\n")
    }

    func toggleRecording() {
        switch phase {
        case .recording, .simulatingFile:
            stopRecording()
        case .idle, .failed:
            Task {
                await startSession(input: .microphone)
            }
        case .connecting, .finalizing:
            break
        }
    }

    func chooseAudioFileAndSimulateRealtime() {
        guard !phase.isBusy else {
            return
        }

        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.audio]

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        selectedAudioFileName = url.lastPathComponent
        Task {
            await startSession(input: .audioFile(url))
        }
    }

    func copyServerCommand() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(localServerCommand, forType: .string)
    }

    func clearTranscript() {
        lines = []
        bufferTranscription = ""
        bufferDiarization = ""
        bufferTranslation = ""
        remainingTranscription = 0
        remainingDiarization = 0
        audioLevel = 0
        elapsedSeconds = 0
    }

    func copyTranscript() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(transcriptText, forType: .string)
    }

    private func startSession(input: SessionInput) async {
        clearTranscript()
        pendingInput = input
        phase = .connecting
        statusText = "Connecting to \(endpoint)"

        do {
            let url = try websocketURL()
            await transport.connect(
                to: url,
                onMessage: { [weak self] message in
                    Task { @MainActor in
                        await self?.handle(message)
                    }
                },
                onClose: { [weak self] reason in
                    Task { @MainActor in
                        self?.handleClose(reason)
                    }
                }
            )
        } catch {
            fail("Invalid WebSocket URL", detail: error.localizedDescription)
        }
    }

    private func stopRecording() {
        guard phase == .recording || phase == .simulatingFile else {
            return
        }

        fileStreamTask?.cancel()
        fileStreamTask = nil
        audioCapture.stop()
        audioLevel = 0
        phase = .finalizing
        statusText = "Finalizing audio"

        Task {
            do {
                try await transport.sendEndOfAudio()
            } catch {
                await MainActor.run {
                    fail("Could not finalize audio", detail: error.localizedDescription)
                }
            }
        }
    }

    private func finishInputStream() {
        guard phase == .recording || phase == .simulatingFile else {
            return
        }

        audioCapture.stop()
        fileStreamTask?.cancel()
        fileStreamTask = nil
        audioLevel = 0
        phase = .finalizing
        statusText = "Finalizing audio"

        Task {
            do {
                try await transport.sendEndOfAudio()
            } catch {
                await MainActor.run {
                    fail("Could not finalize audio", detail: error.localizedDescription)
                }
            }
        }
    }

    private func handle(_ message: ServerMessage) async {
        switch message {
        case .config(let config):
            await handle(config)
        case .update(let update):
            handle(update)
        case .readyToStop:
            audioCapture.stop()
            fileStreamTask?.cancel()
            fileStreamTask = nil
            stopTimer()
            audioLevel = 0
            phase = .idle
            statusText = "Finished"
            await transport.close()
        }
    }

    private func handle(_ config: ServerConfig) async {
        guard config.useAudioWorklet else {
            fail(
                "PCM mode required",
                detail: "Start the WhisperLiveKit server with --pcm-input for the native macOS frontend."
            )
            await transport.close()
            return
        }

        switch pendingInput {
        case .microphone:
            do {
                try await audioCapture.start { [transport] data, level in
                    Task {
                        await transport.sendAudio(data)
                    }
                    Task { @MainActor [weak self] in
                        self?.audioLevel = level
                    }
                }
                phase = .recording
                statusText = "Recording"
                startTimer()
            } catch {
                fail("Microphone unavailable", detail: error.localizedDescription)
                await transport.close()
            }
        case .audioFile(let url):
            phase = .simulatingFile
            statusText = "Simulating \(url.lastPathComponent)"
            startTimer()

            fileStreamTask = Task { [weak self, transport, fileStreamer] in
                do {
                    try await fileStreamer.streamRealtime(url: url) { data, level in
                        Task {
                            await transport.sendAudio(data)
                        }
                        Task { @MainActor [weak self] in
                            self?.audioLevel = level
                        }
                    }
                    await MainActor.run {
                        self?.finishInputStream()
                    }
                } catch is CancellationError {
                    return
                } catch {
                    await MainActor.run {
                        self?.fail("Could not stream audio file", detail: error.localizedDescription)
                    }
                }
            }
        }
    }

    private func handle(_ update: TranscriptionUpdate) {
        if let error = update.error, !error.isEmpty {
            fail("Server error", detail: error)
            return
        }

        if update.status == "no_audio_detected" {
            statusText = "Listening"
        } else if phase == .recording {
            statusText = "Recording"
        }

        lines = update.lines ?? []
        bufferTranscription = update.bufferTranscription ?? ""
        bufferDiarization = update.bufferDiarization ?? ""
        bufferTranslation = update.bufferTranslation ?? ""
        remainingTranscription = update.remainingTimeTranscription ?? 0
        remainingDiarization = update.remainingTimeDiarization ?? 0
    }

    private func handleClose(_ reason: String) {
        audioCapture.stop()
        fileStreamTask?.cancel()
        fileStreamTask = nil
        stopTimer()
        audioLevel = 0

        switch phase {
        case .idle:
            break
        case .finalizing:
            phase = .idle
            statusText = "Finished"
        default:
            fail("Connection closed", detail: reason)
        }
    }

    private func websocketURL() throws -> URL {
        guard var components = URLComponents(string: endpoint), components.scheme == "ws" || components.scheme == "wss" else {
            throw URLError(.badURL)
        }

        var queryItems = components.queryItems ?? []
        if language != "server" {
            queryItems.removeAll { $0.name == "language" }
            queryItems.append(URLQueryItem(name: "language", value: language))
        }
        queryItems.removeAll { $0.name == "mode" }
        queryItems.append(URLQueryItem(name: "mode", value: "full"))
        components.queryItems = queryItems

        guard let url = components.url else {
            throw URLError(.badURL)
        }

        return url
    }

    private var serverLanguage: String {
        language == "server" ? "auto" : language
    }

    private func startTimer() {
        startedAt = Date()
        elapsedSeconds = 0
        timerTask?.cancel()
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                await MainActor.run {
                    guard let self, let startedAt = self.startedAt else {
                        return
                    }
                    self.elapsedSeconds = Date().timeIntervalSince(startedAt)
                }
            }
        }
    }

    private func stopTimer() {
        timerTask?.cancel()
        timerTask = nil
        startedAt = nil
    }

    private func fail(_ title: String, detail: String) {
        audioCapture.stop()
        stopTimer()
        audioLevel = 0
        phase = .failed(title)
        statusText = title
        alert = AppAlert(title: title, message: detail)
    }
}

private enum SessionInput {
    case microphone
    case audioFile(URL)
}
