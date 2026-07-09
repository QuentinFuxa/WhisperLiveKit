import SwiftUI

struct ContentView: View {
    @StateObject private var model = TranscriptionViewModel()

    var body: some View {
        NavigationSplitView {
            SettingsSidebar(model: model)
                .navigationSplitViewColumnWidth(min: 300, ideal: 320, max: 380)
        } detail: {
            TranscriptSurface(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationSplitViewStyle(.balanced)
        .background(Color.wlkBackground)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    model.copyTranscript()
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .disabled(model.transcriptText.isEmpty)

                Button {
                    model.clearTranscript()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .disabled(model.phase.isBusy || model.transcriptText.isEmpty)
            }
        }
        .alert(item: $model.alert) { alert in
            Alert(
                title: Text(alert.title),
                message: Text(alert.message),
                dismissButton: .default(Text("OK"))
            )
        }
        .onReceive(NotificationCenter.default.publisher(for: .clearTranscriptRequested)) { _ in
            if !model.phase.isBusy {
                model.clearTranscript()
            }
        }
    }
}

private struct SettingsSidebar: View {
    @ObservedObject var model: TranscriptionViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Settings")
                    .font(.title2.weight(.semibold))

                SettingsGroup {
                    TextField("ws://localhost:8000/asr", text: $model.endpoint)
                        .textFieldStyle(.roundedBorder)
                        .disabled(model.phase.isBusy)
                } label: {
                    SettingsLabel("Server", systemImage: "network")
                }

                SettingsGroup {
                    Picker("", selection: $model.modelFamily) {
                        ForEach(model.modelFamilies) { family in
                            Text(family.title).tag(family.id)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .disabled(model.phase.isBusy)
                } label: {
                    SettingsLabel("Model family", systemImage: "square.stack.3d.up")
                }

                SettingsGroup {
                    Picker("", selection: $model.backend) {
                        ForEach(model.availableBackends) { backend in
                            Text(backend.title).tag(backend.id)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .disabled(model.phase.isBusy)
                } label: {
                    SettingsLabel("Backend", systemImage: "server.rack")
                }

                SettingsGroup {
                    if model.availableModelSizes.isEmpty {
                        Text(model.modelSizeControlTitle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 9)
                            .padding(.vertical, 6)
                            .background(Color.wlkButtonBackground, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .overlay {
                                RoundedRectangle(cornerRadius: 6, style: .continuous)
                                    .stroke(Color.wlkButtonBorder, lineWidth: 1)
                            }
                    } else {
                        Picker("Size", selection: $model.modelSize) {
                            ForEach(model.availableModelSizes) { modelSize in
                                Text(modelSize.title).tag(modelSize.id)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .disabled(model.phase.isBusy)
                    }
                } label: {
                    SettingsLabel("Size", systemImage: "cpu")
                }

                SettingsGroup {
                    if model.availableStreamingStrategies.count > 1 {
                        Picker("", selection: $model.streamingStrategy) {
                            ForEach(model.availableStreamingStrategies) { strategy in
                                Text(strategy.title).tag(strategy.id)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .disabled(model.phase.isBusy)
                    } else {
                        Picker("", selection: $model.streamingStrategy) {
                            ForEach(model.availableStreamingStrategies) { strategy in
                                Text(strategy.title).tag(strategy.id)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .disabled(true)
                    }
                } label: {
                    SettingsLabel("Streaming strategy", systemImage: "timeline.selection")
                }

                SettingsGroup {
                    Picker("", selection: $model.language) {
                        ForEach(model.languages, id: \.id) { language in
                            Text(language.title).tag(language.id)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .disabled(model.phase.isBusy)
                } label: {
                    SettingsLabel("Language", systemImage: "globe")
                }

                SettingsGroup {
                    Toggle("Diarization", isOn: $model.diarization)
                        .disabled(model.phase.isBusy)
                    Toggle("Punctuation split", isOn: $model.punctuationSplit)
                        .disabled(model.phase.isBusy || !model.diarization)
                } label: {
                    SettingsLabel("Options", systemImage: "switch.2")
                }

                SettingsGroup {
                    HStack(alignment: .top, spacing: 8) {
                        Text(model.localServerCommand)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .lineLimit(4)
                            .fixedSize(horizontal: false, vertical: true)

                        Spacer(minLength: 0)

                        Button {
                            model.copyServerCommand()
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(.borderless)
                        .help("Copy")
                    }
                } label: {
                    SettingsLabel("Local command", systemImage: "terminal")
                }

                Spacer(minLength: 0)
            }
            .padding(20)
        }
    }
}

private struct SettingsGroup<Label: View, Content: View>: View {
    @ViewBuilder let content: Content
    @ViewBuilder let label: Label

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            label
            content
        }
    }
}

private struct SettingsLabel: View {
    let title: String
    let systemImage: String

    init(_ title: String, systemImage: String) {
        self.title = title
        self.systemImage = systemImage
    }

    var body: some View {
        Label(title, systemImage: systemImage)
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
    }
}

private struct TranscriptSurface: View {
    @ObservedObject var model: TranscriptionViewModel

    var body: some View {
        VStack(spacing: 0) {
            HeaderControls(model: model)
                .padding(.top, 18)
                .padding(.bottom, 16)

            Divider()

            TranscriptScroll(model: model)
        }
    }
}

private struct HeaderControls: View {
    @ObservedObject var model: TranscriptionViewModel

    var body: some View {
        Group {
            if #available(macOS 26.0, *) {
                GlassEffectContainer(spacing: 14) {
                    content
                }
            } else {
                content
            }
        }
    }

    private var content: some View {
        VStack(spacing: 12) {
            HStack(spacing: 14) {
                RecordButton(model: model)

                if model.phase == .simulatingFile {
                    FileSimulationPauseButton(model: model)
                }

                FileButton {
                    model.chooseAudioFileAndSimulateRealtime()
                }
                .disabled(model.phase.isBusy)
            }

            Text(model.statusText)
                .font(.system(size: 16))
                .foregroundStyle(.primary)

            if let fileName = model.selectedAudioFileName, model.phase == .simulatingFile {
                Text(fileName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
    }
}

private struct RecordButton: View {
    @ObservedObject var model: TranscriptionViewModel

    var body: some View {
        Button {
            model.toggleRecording()
        } label: {
            HStack(spacing: model.isInputActive ? 14 : 0) {
                RoundedRectangle(cornerRadius: model.isInputActive ? 5 : 999, style: .continuous)
                    .fill(model.phase == .finalizing ? Color.gray : Color(red: 0.82, green: 0.24, blue: 0.21))
                    .frame(width: 25, height: 25)

                if model.isInputActive {
                    MiniWaveform(level: model.audioLevel)
                        .frame(width: 64, height: 22)

                    Text(formattedElapsed)
                        .font(.system(size: 14, weight: .medium))
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                }
            }
            .frame(width: model.isInputActive ? 180 : 50, height: 50)
            .systemGlass(in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(model.phase == .connecting || model.phase == .finalizing)
        .scaleEffect(model.isInputActive ? 1 : 0.98)
        .animation(.smooth(duration: 0.24), value: model.phase)
        .help(model.isInputActive ? "Stop" : "Record")
    }

    private var formattedElapsed: String {
        let seconds = Int(model.elapsedSeconds)
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

private struct FileButton: View {
    let action: () -> Void

    var body: some View {
        Group {
            if #available(macOS 26.0, *) {
                Button(action: action) {
                    Image(systemName: "doc.badge.plus")
                        .font(.system(size: 18, weight: .medium))
                }
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .controlSize(.extraLarge)
            } else {
                Button(action: action) {
                    Image(systemName: "doc.badge.plus")
                        .font(.system(size: 18, weight: .medium))
                        .frame(width: 42, height: 42)
                        .systemGlass(in: Circle())
                }
                .buttonStyle(.plain)
            }
        }
        .help("Load audio file")
        .accessibilityLabel("Load audio file")
    }
}

private struct FileSimulationPauseButton: View {
    @ObservedObject var model: TranscriptionViewModel

    var body: some View {
        Group {
            if #available(macOS 26.0, *) {
                Button {
                    model.toggleFileSimulationPause()
                } label: {
                    Image(systemName: model.fileSimulationPauseSystemImage)
                        .font(.system(size: 16, weight: .medium))
                }
                .buttonStyle(.glass)
                .buttonBorderShape(.circle)
                .controlSize(.extraLarge)
            } else {
                Button {
                    model.toggleFileSimulationPause()
                } label: {
                    Image(systemName: model.fileSimulationPauseSystemImage)
                        .font(.system(size: 16, weight: .medium))
                        .frame(width: 42, height: 42)
                        .systemGlass(in: Circle())
                }
                .buttonStyle(.plain)
            }
        }
        .disabled(model.phase != .simulatingFile)
        .help(model.fileSimulationPauseAccessibilityLabel)
        .accessibilityLabel(model.fileSimulationPauseAccessibilityLabel)
    }
}

private struct TranscriptScroll: View {
    @ObservedObject var model: TranscriptionViewModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if model.lines.isEmpty && pendingText.isEmpty {
                        ContentUnavailableView(
                            "No transcript yet",
                            systemImage: "text.bubble",
                            description: Text(model.statusText)
                        )
                        .frame(maxWidth: .infinity, minHeight: 360)
                    } else {
                        ForEach(Array(model.lines.enumerated()), id: \.element.id) { index, line in
                            TranscriptLineView(
                                line: line,
                                isLast: index == model.lines.count - 1,
                                isFinalizing: model.phase == .finalizing,
                                transcriptionProcessingLag: model.remainingTranscriptionProcessing,
                                transcriptionPolicyLag: model.remainingTranscriptionPolicy,
                                diarizationLag: model.remainingDiarization
                            )
                            .id(line.id)
                        }

                        if !pendingText.isEmpty {
                            PendingTranscriptView(
                                text: pendingText,
                                translation: model.bufferTranslation,
                                isActive: model.isInputActive
                            )
                            .id("pending")
                        }
                    }
                }
                .frame(maxWidth: 700, alignment: .leading)
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .center)
            }
            .scrollIndicators(.hidden)
            .background(Color.wlkBackground)
            .onChange(of: model.lines.count) { _, _ in
                scrollToBottom(proxy)
            }
            .onChange(of: pendingText) { _, _ in
                scrollToBottom(proxy)
            }
        }
    }

    private var pendingText: String {
        [model.bufferDiarization, model.bufferTranscription]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        withAnimation(.smooth(duration: 0.2)) {
            if !pendingText.isEmpty {
                proxy.scrollTo("pending", anchor: .bottom)
            } else if let last = model.lines.last {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        }
    }
}

private struct TranscriptLineView: View {
    let line: TranscriptLine
    let isLast: Bool
    let isFinalizing: Bool
    let transcriptionProcessingLag: Double
    let transcriptionPolicyLag: Double
    let diarizationLag: Double
    private let lagDisplayThreshold = 0.1

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .center, spacing: 10) {
                if line.speaker == -2 {
                    Chip(systemImage: "speaker.slash.fill", text: timeText.nilIfEmpty ?? "Silence", style: .muted)
                } else {
                    Chip(systemImage: "person.fill", text: speakerText, style: .plain)

                    if !timeText.isEmpty {
                        Text(timeText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    if let language = line.detectedLanguage, !language.isEmpty {
                        Chip(systemImage: "globe", text: language, style: .muted)
                    }

                    if isLast && !isFinalizing && transcriptionProcessingLag > lagDisplayThreshold {
                        Chip(systemImage: "cpu", text: "Compute \(String(format: "%.1f", transcriptionProcessingLag))s", style: .active)
                    }

                    if isLast && !isFinalizing && transcriptionPolicyLag > lagDisplayThreshold {
                        Chip(systemImage: "timer", text: "Policy \(String(format: "%.1f", transcriptionPolicyLag))s", style: .muted)
                    }

                    if isLast && !isFinalizing && diarizationLag > 0 {
                        Chip(systemImage: nil, text: "Diarization \(String(format: "%.1f", diarizationLag))s", style: .muted)
                    }
                }
            }

            if line.speaker != -2 {
                Text(line.text?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty ?? "...")
                    .font(.system(size: 16))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .padding(.leading, 10)
                    .padding(.bottom, 10)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if let translation = line.translation?.trimmingCharacters(in: .whitespacesAndNewlines), !translation.isEmpty {
                TranslationChip(text: translation)
                    .padding(.leading, 10)
                    .padding(.bottom, 8)
            }
        }
    }

    private var speakerText: String {
        "Speaker \(line.speaker)"
    }

    private var timeText: String {
        [line.start, line.end].compactMap { $0 }.joined(separator: " - ")
    }
}

private struct PendingTranscriptView: View {
    let text: String
    let translation: String
    let isActive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                if isActive {
                    ProgressView()
                        .controlSize(.small)
                }
                Chip(systemImage: nil, text: "Live", style: .active)
            }

            Text(text)
                .font(.system(size: 16))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .padding(.leading, 10)
                .padding(.bottom, 10)

            let cleanTranslation = translation.trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleanTranslation.isEmpty {
                TranslationChip(text: cleanTranslation)
                    .padding(.leading, 10)
                    .padding(.bottom, 8)
            }
        }
    }
}

private struct TranslationChip: View {
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 4) {
            Image(systemName: "translate")
                .font(.caption2)
                .padding(.top, 2)
            Text(text)
                .font(.system(size: 14))
                .textSelection(.enabled)
        }
        .foregroundStyle(.primary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.wlkChipBackground, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct Chip: View {
    enum Style {
        case plain
        case muted
        case active
    }

    let systemImage: String?
    let text: String
    let style: Style

    var body: some View {
        HStack(spacing: 5) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.caption2)
            }
            Text(text)
                .font(.system(size: 14))
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, 10)
        .padding(.vertical, 2)
        .background(background, in: Capsule())
        .overlay {
            if style == .plain {
                Capsule().stroke(Color.wlkBorder, lineWidth: 1)
            }
        }
    }

    private var foreground: Color {
        switch style {
        case .plain:
            return .primary
        case .muted:
            return .secondary
        case .active:
            return Color.wlkText
        }
    }

    private var background: Color {
        switch style {
        case .plain:
            return .clear
        case .muted:
            return Color.wlkMutedChipBackground
        case .active:
            return Color.wlkLoadingBackground
        }
    }
}

private struct MiniWaveform: View {
    let level: Float

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            GeometryReader { geometry in
                let sampleCount = 44
                let width = max(geometry.size.width, 1)
                let height = max(geometry.size.height, 1)
                let centerY = height * 0.5
                let clampedLevel = min(max(CGFloat(level), 0), 1)
                let amplitude = height * max(0.015, clampedLevel * 0.44)
                let phase = timeline.date.timeIntervalSinceReferenceDate * 7.0

                Path { path in
                    for index in 0..<sampleCount {
                        let progress = CGFloat(index) / CGFloat(sampleCount - 1)
                        let x = progress * width
                        let envelope = 0.62 + 0.38 * sin(progress * .pi)
                        let y = centerY + sin(progress * .pi * 4.0 + phase) * amplitude * envelope

                        if index == 0 {
                            path.move(to: CGPoint(x: x, y: y))
                        } else {
                            path.addLine(to: CGPoint(x: x, y: y))
                        }
                    }
                }
                .stroke(Color.wlkText, style: StrokeStyle(lineWidth: 1.25, lineCap: .round, lineJoin: .round))
            }
        }
    }
}

private extension Color {
    static let wlkBackground = Color(nsColor: .windowBackgroundColor)
    static let wlkButtonBackground = Color(nsColor: .controlBackgroundColor)
    static let wlkButtonBorder = Color(nsColor: .separatorColor).opacity(0.55)
    static let wlkBorder = Color(nsColor: .separatorColor).opacity(0.5)
    static let wlkChipBackground = Color.primary.opacity(0.045)
    static let wlkMutedChipBackground = Color.primary.opacity(0.055)
    static let wlkLoadingBackground = Color(red: 1.0, green: 0.30, blue: 0.30).opacity(0.08)
    static let wlkText = Color.primary
}

private extension View {
    @ViewBuilder
    func systemGlass<S: Shape>(in shape: S) -> some View {
        if #available(macOS 26.0, *) {
            glassEffect(.regular.interactive(true), in: shape)
        } else {
            background(.regularMaterial, in: shape)
                .overlay {
                    shape.stroke(Color.wlkButtonBorder, lineWidth: 1)
                }
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}
