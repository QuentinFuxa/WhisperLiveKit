import Foundation

actor WebSocketTransport {
    private var task: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private let decoder = JSONDecoder()

    func connect(
        to url: URL,
        onMessage: @escaping @Sendable (ServerMessage) -> Void,
        onClose: @escaping @Sendable (String) -> Void
    ) {
        close()

        let task = URLSession.shared.webSocketTask(with: url)
        self.task = task
        task.resume()

        receiveTask = Task { [weak self] in
            await self?.receiveLoop(task: task, onMessage: onMessage, onClose: onClose)
        }
    }

    func sendAudio(_ data: Data) async {
        guard !data.isEmpty, let task else {
            return
        }

        do {
            try await task.send(.data(data))
        } catch {
            // The receive loop owns user-facing connection state.
        }
    }

    func sendEndOfAudio() async throws {
        guard let task else {
            return
        }
        try await task.send(.data(Data()))
    }

    func close() {
        receiveTask?.cancel()
        receiveTask = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    private func receiveLoop(
        task: URLSessionWebSocketTask,
        onMessage: @escaping @Sendable (ServerMessage) -> Void,
        onClose: @escaping @Sendable (String) -> Void
    ) async {
        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                let data: Data

                switch message {
                case .string(let text):
                    data = Data(text.utf8)
                case .data(let payload):
                    data = payload
                @unknown default:
                    continue
                }

                onMessage(try ServerMessage.decode(from: data, using: decoder))
            } catch {
                if !Task.isCancelled {
                    onClose(error.localizedDescription)
                }
                return
            }
        }
    }
}
