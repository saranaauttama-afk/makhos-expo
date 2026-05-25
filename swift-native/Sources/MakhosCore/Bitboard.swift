import Foundation

public typealias BB = UInt32

@inlinable
public func B1(_ index: Int) -> BB {
    BB(1) << index
}

public struct RC: Equatable, Sendable {
    public let r: Int
    public let c: Int

    public init(r: Int, c: Int) {
        self.r = r
        self.c = c
    }
}

public enum Direction: Int, CaseIterable, Sendable {
    case ul = 0
    case ur = 1
    case dl = 2
    case dr = 3
}

public struct Step: Equatable, Sendable {
    public let to: Int
    public let dir: Direction

    public init(to: Int, dir: Direction) {
        self.to = to
        self.dir = dir
    }
}

public struct Jump: Equatable, Sendable {
    public let over: Int
    public let to: Int
    public let dir: Direction

    public init(over: Int, to: Int, dir: Direction) {
        self.over = over
        self.to = to
        self.dir = dir
    }
}

private let directionOffsets: [(Direction, Int, Int)] = [
    (.ul, -1, -1),
    (.ur, -1, 1),
    (.dl, 1, -1),
    (.dr, 1, 1),
]

public let squareToRC: [RC] = {
    var mapping: [RC] = []
    mapping.reserveCapacity(32)
    for r in 0..<8 {
        for c in 0..<8 where ((r + c) & 1) == 1 {
            mapping.append(RC(r: r, c: c))
        }
    }
    return mapping
}()

public let rcToIndex: [Int] = {
    var mapping = Array(repeating: -1, count: 64)
    for (index, rc) in squareToRC.enumerated() {
        mapping[(rc.r * 8) + rc.c] = index
    }
    return mapping
}()

@inlinable
public func toRC(_ index: Int) -> RC {
    squareToRC[index]
}

@inlinable
public func toIndex(r: Int, c: Int) -> Int {
    guard (0..<8).contains(r), (0..<8).contains(c) else { return -1 }
    return rcToIndex[(r * 8) + c]
}

public let steps: [[Step]] = {
    var output = Array(repeating: [Step](), count: 32)
    for index in 0..<32 {
        let rc = toRC(index)
        for (dir, dr, dc) in directionOffsets {
            let next = toIndex(r: rc.r + dr, c: rc.c + dc)
            if next >= 0 {
                output[index].append(Step(to: next, dir: dir))
            }
        }
    }
    return output
}()

public let jumps: [[Jump]] = {
    var output = Array(repeating: [Jump](), count: 32)
    for index in 0..<32 {
        let rc = toRC(index)
        for (dir, dr, dc) in directionOffsets {
            let over = toIndex(r: rc.r + dr, c: rc.c + dc)
            let landing = toIndex(r: rc.r + (2 * dr), c: rc.c + (2 * dc))
            if over >= 0, landing >= 0 {
                output[index].append(Jump(over: over, to: landing, dir: dir))
            }
        }
    }
    return output
}()

public let nextSquares: [[Int]] = {
    var output = Array(repeating: Array(repeating: -1, count: 4), count: 32)
    for square in 0..<32 {
        let rc = toRC(square)
        for (dir, dr, dc) in directionOffsets {
            let next = toIndex(r: rc.r + dr, c: rc.c + dc)
            if next >= 0 {
                output[square][dir.rawValue] = next
            }
        }
    }
    return output
}()

public let rays: [[[Int]]] = {
    var output = Array(
        repeating: Array(repeating: [Int](), count: 4),
        count: 32
    )
    for square in 0..<32 {
        let rc = toRC(square)
        for (dir, dr, dc) in directionOffsets {
            var nr = rc.r + dr
            var nc = rc.c + dc
            while (0..<8).contains(nr), (0..<8).contains(nc) {
                let target = toIndex(r: nr, c: nc)
                if target >= 0 {
                    output[square][dir.rawValue].append(target)
                }
                nr += dr
                nc += dc
            }
        }
    }
    return output
}()

@inlinable
public func bitCount(_ value: BB) -> Int {
    value.nonzeroBitCount
}

@inlinable
public func bits(_ value: BB) -> [Int] {
    var x = value
    var output: [Int] = []
    output.reserveCapacity(value.nonzeroBitCount)
    while x != 0 {
        let bit = x.trailingZeroBitCount
        output.append(bit)
        x &= x &- 1
    }
    return output
}
