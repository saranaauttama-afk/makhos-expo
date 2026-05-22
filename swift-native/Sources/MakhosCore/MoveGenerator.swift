public enum MoveGenerator {
    private static let p1DirA = Direction.ul.rawValue
    private static let p1DirB = Direction.ur.rawValue
    private static let p2DirA = Direction.dl.rawValue
    private static let p2DirB = Direction.dr.rawValue
    private static let lastRankP1: BB = 0b00000000000000000000000000001111
    private static let lastRankP2: BB = 0b11110000000000000000000000000000
    private static let maxChain = 16

    static func willPromote(_ side: Side, to: Int) -> Bool {
        let bit = B1(to)
        return side == .playerOne ? (lastRankP1 & bit) != 0 : (lastRankP2 & bit) != 0
    }

    public static func applyMove(_ position: Position, _ move: Move) -> Position {
        let fromBit = B1(move.from)
        let toBit = B1(move.to)
        let isCapture = !move.captured.isEmpty

        var p1Men = position.p1Men
        var p1Kings = position.p1Kings
        var p2Men = position.p2Men
        var p2Kings = position.p2Kings

        if position.side == .playerOne {
            let movingKing = (p1Kings & fromBit) != 0
            if movingKing {
                p1Kings = (p1Kings & ~fromBit) | toBit
            } else {
                p1Men = (p1Men & ~fromBit) | toBit
            }

            for captured in move.captured {
                let captureBit = B1(captured)
                if (p2Men & captureBit) != 0 {
                    p2Men &= ~captureBit
                } else {
                    p2Kings &= ~captureBit
                }
            }

            if move.promote && !movingKing {
                p1Men &= ~toBit
                p1Kings |= toBit
            }
        } else {
            let movingKing = (p2Kings & fromBit) != 0
            if movingKing {
                p2Kings = (p2Kings & ~fromBit) | toBit
            } else {
                p2Men = (p2Men & ~fromBit) | toBit
            }

            for captured in move.captured {
                let captureBit = B1(captured)
                if (p1Men & captureBit) != 0 {
                    p1Men &= ~captureBit
                } else {
                    p1Kings &= ~captureBit
                }
            }

            if move.promote && !movingKing {
                p2Men &= ~toBit
                p2Kings |= toBit
            }
        }

        return Position(
            side: position.side.opponent,
            p1Men: p1Men,
            p1Kings: p1Kings,
            p2Men: p2Men,
            p2Kings: p2Kings,
            halfmoveClock: isCapture ? 0 : position.halfmoveClock + 1
        )
    }

    public static func generateMoves(_ position: Position) -> [Move] {
        var output: [Move] = []
        generateMoves(position, into: &output)
        return output
    }

    public static func generateMoves(_ position: Position, into output: inout [Move]) {
        generateCaptures(position, into: &output)
        if !output.isEmpty { return }

        let occ = occupied(position)
        let empty = ~occ
        let men = sideMen(position)
        let kings = sideKings(position)

        if position.side == .playerOne {
            addMenQuietMoves(
                side: position.side,
                men: men,
                empty: empty,
                output: &output,
                dirA: p1DirA,
                dirB: p1DirB
            )
        } else {
            addMenQuietMoves(
                side: position.side,
                men: men,
                empty: empty,
                output: &output,
                dirA: p2DirA,
                dirB: p2DirB
            )
        }

        addKingQuietMoves(kings: kings, occupied: occ, output: &output)
    }

    public static func generateCaptures(_ position: Position) -> [Move] {
        var output: [Move] = []
        generateCaptures(position, into: &output)
        return output
    }

    public static func generateCaptures(_ position: Position, into output: inout [Move]) {
        output.removeAll(keepingCapacity: true)
        let men = sideMen(position)
        let kings = sideKings(position)

        for from in bits(men) {
            genMenCaptures(position: position, from: from, output: &output)
        }
        for from in bits(kings) {
            genKingCaptures(position: position, from: from, output: &output)
        }

        guard output.count > 1 else { return }
        let maxCaptures = output.map(\.captured.count).max() ?? 0
        output = output.filter { $0.captured.count == maxCaptures }
    }

    public static func hasCapturesAvailable(_ position: Position) -> Bool {
        let occ = occupied(position)
        let myMen = sideMen(position)
        let myKings = sideKings(position)
        let myAll = myMen | myKings
        let opAll = position.side == .playerOne
            ? (position.p2Men | position.p2Kings)
            : (position.p1Men | position.p1Kings)
        let empty = ~occ

        let dirA = position.side == .playerOne ? p1DirA : p2DirA
        let dirB = position.side == .playerOne ? p1DirB : p2DirB

        for from in bits(myMen) {
            if hasMenCapture(from: from, dir: dirA, opponent: opAll, empty: empty) { return true }
            if hasMenCapture(from: from, dir: dirB, opponent: opAll, empty: empty) { return true }
        }

        for from in bits(myKings) {
            for dir in 0..<4 {
                let ray = rays[from][dir]
                var seenEnemy = false
                for square in ray {
                    let bit = B1(square)
                    if (myAll & bit) != 0 { break }
                    if (opAll & bit) != 0 {
                        if seenEnemy { break }
                        seenEnemy = true
                        continue
                    }
                    if seenEnemy { return true }
                }
            }
        }

        return false
    }

    private static func addMenQuietMoves(
        side: Side,
        men: BB,
        empty: BB,
        output: inout [Move],
        dirA: Int,
        dirB: Int
    ) {
        for from in bits(men) {
            let toA = nextSquares[from][dirA]
            if toA >= 0, (empty & B1(toA)) != 0 {
                output.append(Move(from: from, to: toA, promote: willPromote(side, to: toA)))
            }
            let toB = nextSquares[from][dirB]
            if toB >= 0, (empty & B1(toB)) != 0 {
                output.append(Move(from: from, to: toB, promote: willPromote(side, to: toB)))
            }
        }
    }

    private static func addKingQuietMoves(kings: BB, occupied: BB, output: inout [Move]) {
        for from in bits(kings) {
            for dir in 0..<4 {
                for to in rays[from][dir] {
                    if (occupied & B1(to)) != 0 { break }
                    output.append(Move(from: from, to: to))
                }
            }
        }
    }

    private static func hasMenCapture(from: Int, dir: Int, opponent: BB, empty: BB) -> Bool {
        let over = nextSquares[from][dir]
        if over < 0 || (opponent & B1(over)) == 0 { return false }
        let landing = nextSquares[over][dir]
        if landing < 0 { return false }
        return (empty & B1(landing)) != 0
    }

    private static func genMenCaptures(position: Position, from: Int, output: inout [Move]) {
        var captures: [Int] = []
        var path: [Int] = []

        let myMen0 = position.side == .playerOne ? position.p1Men : position.p2Men
        let myKings0 = position.side == .playerOne ? position.p1Kings : position.p2Kings
        let opMen0 = position.side == .playerOne ? position.p2Men : position.p1Men
        let opKings0 = position.side == .playerOne ? position.p2Kings : position.p1Kings
        let dirA = position.side == .playerOne ? p1DirA : p2DirA
        let dirB = position.side == .playerOne ? p1DirB : p2DirB

        func dfs(cur: Int, myMen: BB, myKings: BB, opMen: BB, opKings: BB) {
            var extended = false
            if tryDir(cur: cur, dir: dirA, myMen: myMen, myKings: myKings, opMen: opMen, opKings: opKings) {
                extended = true
            }
            if tryDir(cur: cur, dir: dirB, myMen: myMen, myKings: myKings, opMen: opMen, opKings: opKings) {
                extended = true
            }

            if !extended, let to = path.last {
                output.append(Move(
                    from: from,
                    to: to,
                    captured: captures,
                    promote: willPromote(position.side, to: to),
                    path: path
                ))
            }
        }

        @discardableResult
        func tryDir(cur: Int, dir: Int, myMen: BB, myKings: BB, opMen: BB, opKings: BB) -> Bool {
            if captures.count >= maxChain { return false }

            let over = nextSquares[cur][dir]
            if over < 0 { return false }
            let overBit = B1(over)
            if ((opMen | opKings) & overBit) == 0 { return false }

            let landing = nextSquares[over][dir]
            if landing < 0 { return false }
            let landingBit = B1(landing)
            let occupiedNow = myMen | myKings | opMen | opKings
            if (occupiedNow & landingBit) != 0 { return false }

            let fromBit = B1(cur)
            var nextMyMen = myMen
            var nextMyKings = myKings
            var nextOpMen = opMen
            var nextOpKings = opKings

            if (nextMyKings & fromBit) != 0 {
                nextMyKings = (nextMyKings & ~fromBit) | landingBit
            } else {
                nextMyMen = (nextMyMen & ~fromBit) | landingBit
            }

            if (nextOpKings & overBit) != 0 {
                nextOpKings &= ~overBit
            } else {
                nextOpMen &= ~overBit
            }

            captures.append(over)
            path.append(landing)
            dfs(cur: landing, myMen: nextMyMen, myKings: nextMyKings, opMen: nextOpMen, opKings: nextOpKings)
            path.removeLast()
            captures.removeLast()
            return true
        }

        dfs(cur: from, myMen: myMen0, myKings: myKings0, opMen: opMen0, opKings: opKings0)
    }

    private static func genKingCaptures(position: Position, from: Int, output: inout [Move]) {
        let myKings0 = position.side == .playerOne ? position.p1Kings : position.p2Kings
        guard (myKings0 & B1(from)) != 0 else { return }

        var captures: [Int] = []
        var path: [Int] = []

        let myMen0 = position.side == .playerOne ? position.p1Men : position.p2Men
        let opMen0 = position.side == .playerOne ? position.p2Men : position.p1Men
        let opKings0 = position.side == .playerOne ? position.p2Kings : position.p1Kings

        func dfs(cur: Int, myMen: BB, myKings: BB, opMen: BB, opKings: BB) {
            var extended = false
            for dir in 0..<4 where tryDir(cur: cur, dir: dir, myMen: myMen, myKings: myKings, opMen: opMen, opKings: opKings) {
                extended = true
            }

            if !extended, let to = path.last {
                output.append(Move(from: from, to: to, captured: captures, promote: false, path: path))
            }
        }

        @discardableResult
        func tryDir(cur: Int, dir: Int, myMen: BB, myKings: BB, opMen: BB, opKings: BB) -> Bool {
            if captures.count >= maxChain { return false }

            let ray = rays[cur][dir]
            var enemy = -1

            for square in ray {
                let bit = B1(square)
                if ((myMen | myKings) & bit) != 0 { return false }

                if enemy < 0 {
                    if ((opMen | opKings) & bit) != 0 {
                        enemy = square
                    }
                    continue
                }

                if ((myMen | myKings | opMen | opKings) & bit) != 0 { return false }

                let landing = square
                let fromBit = B1(cur)
                let landingBit = B1(landing)
                let enemyBit = B1(enemy)

                let nextMyMen = myMen
                var nextMyKings = myKings
                var nextOpMen = opMen
                var nextOpKings = opKings

                nextMyKings = (nextMyKings & ~fromBit) | landingBit
                if (nextOpKings & enemyBit) != 0 {
                    nextOpKings &= ~enemyBit
                } else {
                    nextOpMen &= ~enemyBit
                }

                captures.append(enemy)
                path.append(landing)
                dfs(cur: landing, myMen: nextMyMen, myKings: nextMyKings, opMen: nextOpMen, opKings: nextOpKings)
                path.removeLast()
                captures.removeLast()
                return true
            }

            return false
        }

        dfs(cur: from, myMen: myMen0, myKings: myKings0, opMen: opMen0, opKings: opKings0)
    }
}

@inlinable
public func applyMove(_ position: Position, _ move: Move) -> Position {
    MoveGenerator.applyMove(position, move)
}

@inlinable
public func generateMoves(_ position: Position) -> [Move] {
    MoveGenerator.generateMoves(position)
}

@inlinable
public func generateCaptures(_ position: Position) -> [Move] {
    MoveGenerator.generateCaptures(position)
}

@inlinable
public func hasCapturesAvailable(_ position: Position) -> Bool {
    MoveGenerator.hasCapturesAvailable(position)
}
