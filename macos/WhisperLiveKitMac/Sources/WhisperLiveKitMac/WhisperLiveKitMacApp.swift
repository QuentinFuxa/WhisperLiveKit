import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@main
struct WhisperLiveKitMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 980, minHeight: 640)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unifiedCompact)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Clear Transcript") {
                    NotificationCenter.default.post(name: .clearTranscriptRequested, object: nil)
                }
                .keyboardShortcut("k", modifiers: [.command])
            }
        }
    }
}

extension Notification.Name {
    static let clearTranscriptRequested = Notification.Name("WhisperLiveKitClearTranscriptRequested")
}
