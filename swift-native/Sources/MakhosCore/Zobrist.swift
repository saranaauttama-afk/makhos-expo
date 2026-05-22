private struct Mulberry32 {
    private var state: UInt32

    init(seed: UInt32) {
        state = seed
    }

    mutating func nextUInt32() -> UInt32 {
        state &+= 0x6D2B79F5
        var t = state
        t = (t ^ (t >> 15)) &* (1 | t)
        t ^= t &+ ((t ^ (t >> 7)) &* (61 | t))
        return t ^ (t >> 14)
    }
}

public enum Zobrist {
    public static let piece: [[UInt32]] = {
        var rng = Mulberry32(seed: 0x00C0FFEE)
        return (0..<4).map { _ in
            (0..<32).map { _ in rng.nextUInt32() }
        }
    }()

    public static let side: UInt32 = {
        var rng = Mulberry32(seed: 0x00C0FFEE)
        _ = (0..<(4 * 32)).map { _ in rng.nextUInt32() }
        return rng.nextUInt32()
    }()

    public static let verifyPiece: [[UInt32]] = {
        var rng = Mulberry32(seed: 0xDEADBEEF)
        return (0..<4).map { _ in
            (0..<32).map { _ in rng.nextUInt32() }
        }
    }()

    public static let verifySide: UInt32 = {
        var rng = Mulberry32(seed: 0xDEADBEEF)
        _ = (0..<(4 * 32)).map { _ in rng.nextUInt32() }
        return rng.nextUInt32()
    }()
}

public func hashPosition(_ position: Position) -> UInt32 {
    var hash: UInt32 = 0
    for index in bits(position.p1Men) { hash ^= Zobrist.piece[0][index] }
    for index in bits(position.p1Kings) { hash ^= Zobrist.piece[1][index] }
    for index in bits(position.p2Men) { hash ^= Zobrist.piece[2][index] }
    for index in bits(position.p2Kings) { hash ^= Zobrist.piece[3][index] }
    if position.side == .playerOne { hash ^= Zobrist.side }
    return hash
}

public func verifyHashPosition(_ position: Position) -> UInt32 {
    var hash: UInt32 = 0
    for index in bits(position.p1Men) { hash ^= Zobrist.verifyPiece[0][index] }
    for index in bits(position.p1Kings) { hash ^= Zobrist.verifyPiece[1][index] }
    for index in bits(position.p2Men) { hash ^= Zobrist.verifyPiece[2][index] }
    for index in bits(position.p2Kings) { hash ^= Zobrist.verifyPiece[3][index] }
    if position.side == .playerOne { hash ^= Zobrist.verifySide }
    return hash
}
