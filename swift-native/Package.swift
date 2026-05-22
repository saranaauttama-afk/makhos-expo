// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "MakhosSwift",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "MakhosCore",
            targets: ["MakhosCore"]
        ),
        .library(
            name: "MakhosNativeUI",
            targets: ["MakhosNativeUI"]
        ),
    ],
    targets: [
        .target(
            name: "MakhosCore"
        ),
        .target(
            name: "MakhosNativeUI",
            dependencies: ["MakhosCore"]
        ),
        .testTarget(
            name: "MakhosCoreTests",
            dependencies: ["MakhosCore"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
