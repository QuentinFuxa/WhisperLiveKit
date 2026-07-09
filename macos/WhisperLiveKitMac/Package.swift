// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "WhisperLiveKitMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "WhisperLiveKitMac", targets: ["WhisperLiveKitMac"])
    ],
    targets: [
        .executableTarget(
            name: "WhisperLiveKitMac",
            path: "Sources/WhisperLiveKitMac",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Resources/Info.plist"
                ])
            ]
        )
    ]
)
