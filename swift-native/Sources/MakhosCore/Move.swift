public struct Move: Equatable, Hashable, Sendable {
    public let from: Int
    public let to: Int
    public let captured: [Int]
    public let promote: Bool
    public let path: [Int]?

    public init(
        from: Int,
        to: Int,
        captured: [Int] = [],
        promote: Bool = false,
        path: [Int]? = nil
    ) {
        self.from = from
        self.to = to
        self.captured = captured
        self.promote = promote
        self.path = path
    }
}
