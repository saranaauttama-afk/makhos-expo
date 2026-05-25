public enum Side: Int, CaseIterable, Sendable {
    case playerOne = 1
    case playerTwo = -1

    public var opponent: Side {
        self == .playerOne ? .playerTwo : .playerOne
    }
}

public struct Position: Equatable, Sendable {
    public var side: Side
    public var p1Men: BB
    public var p1Kings: BB
    public var p2Men: BB
    public var p2Kings: BB
    public var halfmoveClock: Int

    public init(
        side: Side,
        p1Men: BB,
        p1Kings: BB,
        p2Men: BB,
        p2Kings: BB,
        halfmoveClock: Int = 0
    ) {
        self.side = side
        self.p1Men = p1Men
        self.p1Kings = p1Kings
        self.p2Men = p2Men
        self.p2Kings = p2Kings
        self.halfmoveClock = halfmoveClock
    }
}

public let emptyPosition = Position(
    side: .playerOne,
    p1Men: 0,
    p1Kings: 0,
    p2Men: 0,
    p2Kings: 0,
    halfmoveClock: 0
)

public func initialPosition() -> Position {
    var p1Men: BB = 0
    var p2Men: BB = 0

    for square in [0, 1, 2, 3, 4, 5, 6, 7] {
        p2Men |= B1(square)
    }
    for square in [24, 25, 26, 27, 28, 29, 30, 31] {
        p1Men |= B1(square)
    }

    return Position(
        side: .playerOne,
        p1Men: p1Men,
        p1Kings: 0,
        p2Men: p2Men,
        p2Kings: 0,
        halfmoveClock: 0
    )
}

@inlinable
public func occupied(_ position: Position) -> BB {
    (position.p1Men | position.p1Kings | position.p2Men | position.p2Kings)
}

public func isDrawByInactivity(_ position: Position) -> Bool {
    if position.halfmoveClock >= 32 { return true }
    let allKings = position.p1Men == 0 && position.p2Men == 0
    if allKings && position.halfmoveClock >= 16 { return true }
    return false
}

@inlinable
public func sideMen(_ position: Position) -> BB {
    position.side == .playerOne ? position.p1Men : position.p2Men
}

@inlinable
public func sideKings(_ position: Position) -> BB {
    position.side == .playerOne ? position.p1Kings : position.p2Kings
}

@inlinable
public func oppMen(_ position: Position) -> BB {
    position.side == .playerOne ? position.p2Men : position.p1Men
}

@inlinable
public func oppKings(_ position: Position) -> BB {
    position.side == .playerOne ? position.p2Kings : position.p1Kings
}

public func isTerminal(_ position: Position) -> Bool {
    let myCount = bitCount(sideMen(position) | sideKings(position))
    let opponentCount = bitCount(oppMen(position) | oppKings(position))
    return myCount == 0 || opponentCount == 0
}
