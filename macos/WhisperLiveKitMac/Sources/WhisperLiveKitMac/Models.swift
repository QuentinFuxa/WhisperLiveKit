import Foundation

struct ServerConfig: Decodable, Equatable {
    let type: String
    let useAudioWorklet: Bool
    let mode: String?
}

struct TranscriptLine: Decodable, Equatable, Identifiable {
    let speaker: Int
    let text: String?
    let start: String?
    let end: String?
    let translation: String?
    let detectedLanguage: String?

    var id: String {
        "\(speaker)-\(start ?? "")-\(end ?? "")-\(text ?? "")-\(translation ?? "")-\(detectedLanguage ?? "")"
    }

    enum CodingKeys: String, CodingKey {
        case speaker
        case text
        case start
        case end
        case translation
        case detectedLanguage = "detected_language"
    }
}

struct TranscriptionUpdate: Decodable, Equatable {
    let type: String?
    let status: String?
    let lines: [TranscriptLine]?
    let bufferTranscription: String?
    let bufferDiarization: String?
    let bufferTranslation: String?
    let remainingTimeTranscription: Double?
    let remainingTimeTranscriptionProcessing: Double?
    let remainingTimeTranscriptionPolicy: Double?
    let remainingTimeDiarization: Double?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case type
        case status
        case lines
        case bufferTranscription = "buffer_transcription"
        case bufferDiarization = "buffer_diarization"
        case bufferTranslation = "buffer_translation"
        case remainingTimeTranscription = "remaining_time_transcription"
        case remainingTimeTranscriptionProcessing = "remaining_time_transcription_processing"
        case remainingTimeTranscriptionPolicy = "remaining_time_transcription_policy"
        case remainingTimeDiarization = "remaining_time_diarization"
        case error
    }
}

enum ServerMessage {
    case config(ServerConfig)
    case update(TranscriptionUpdate)
    case readyToStop

    static func decode(from data: Data, using decoder: JSONDecoder = JSONDecoder()) throws -> ServerMessage {
        let probe = try decoder.decode(MessageProbe.self, from: data)

        switch probe.type {
        case "config":
            return .config(try decoder.decode(ServerConfig.self, from: data))
        case "ready_to_stop":
            return .readyToStop
        default:
            return .update(try decoder.decode(TranscriptionUpdate.self, from: data))
        }
    }
}

private struct MessageProbe: Decodable {
    let type: String?
}

enum SessionPhase: Equatable {
    case idle
    case connecting
    case recording
    case simulatingFile
    case finalizing
    case failed(String)

    var title: String {
        switch self {
        case .idle:
            return "Ready"
        case .connecting:
            return "Connecting"
        case .recording:
            return "Recording"
        case .simulatingFile:
            return "Simulating"
        case .finalizing:
            return "Finalizing"
        case .failed:
            return "Error"
        }
    }

    var isBusy: Bool {
        switch self {
        case .connecting, .recording, .simulatingFile, .finalizing:
            return true
        case .idle, .failed:
            return false
        }
    }

    var isRecording: Bool {
        self == .recording || self == .simulatingFile
    }
}

struct AppAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}
