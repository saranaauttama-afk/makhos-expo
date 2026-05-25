import MakhosCore
import Testing

private func makePosition(
    side: Side,
    p1Men: BB = 0,
    p1Kings: BB = 0,
    p2Men: BB = 0,
    p2Kings: BB = 0,
    halfmoveClock: Int = 0
) -> Position {
    Position(
        side: side,
        p1Men: p1Men,
        p1Kings: p1Kings,
        p2Men: p2Men,
        p2Kings: p2Kings,
        halfmoveClock: halfmoveClock
    )
}

@Test
func initialPositionHasQuietMovesOnly() {
    let position = initialPosition()
    let moves = generateMoves(position)

    #expect(!moves.isEmpty)
    #expect(hasCapturesAvailable(position) == false)
    #expect(moves.allSatisfy { $0.captured.isEmpty })
}

@Test
func forcedDoubleCaptureMatchesCuratedTypeScriptCase() {
    let position = makePosition(
        side: .playerOne,
        p1Men: B1(22) | B1(30),
        p2Men: B1(17) | B1(9)
    )

    let moves = generateMoves(position)
    #expect(moves.count == 1)
    #expect(moves[0] == Move(from: 22, to: 6, captured: [17, 9], promote: false, path: [13, 6]))
}

@Test
func maxCaptureFilterRemovesShorterChoices() {
    let position = makePosition(
        side: .playerOne,
        p1Men: B1(22) | B1(25),
        p2Men: B1(17) | B1(9) | B1(20)
    )

    let moves = generateMoves(position)
    #expect(!moves.isEmpty)
    #expect(moves.allSatisfy { $0.captured.count == 2 })
}

@Test
func kingFlyCaptureUsesImmediateLandingSquare() {
    let position = makePosition(
        side: .playerOne,
        p1Kings: B1(22),
        p2Men: B1(17)
    )

    let moves = generateMoves(position)
    let found = moves.contains { move in
        move.from == 22 && move.to == 13 && move.captured == [17]
    }
    #expect(found)
}

@Test
func promotionsExistForBothSides() {
    let p1Position = makePosition(
        side: .playerOne,
        p1Men: B1(4),
        p2Men: B1(31)
    )
    let p2Position = makePosition(
        side: .playerTwo,
        p1Men: B1(0),
        p2Men: B1(27)
    )

    let p1Promotes = generateMoves(p1Position).contains { $0.promote }
    let p2Promotes = generateMoves(p2Position).contains { $0.promote }

    #expect(p1Promotes)
    #expect(p2Promotes)
}

@Test
func blockedSideHasNoLegalMove() {
    let position = makePosition(
        side: .playerOne,
        p1Men: B1(0),
        p2Men: B1(4)
    )

    #expect(generateMoves(position).isEmpty)
}

@Test
func applyMoveUpdatesBoardAndHalfmoveClock() {
    let position = makePosition(
        side: .playerOne,
        p1Men: B1(22),
        p2Men: B1(17)
    )
    let move = generateMoves(position)[0]
    let next = applyMove(position, move)

    #expect(next.side == .playerTwo)
    #expect((next.p1Men & B1(13)) != 0)
    #expect((next.p2Men & B1(17)) == 0)
    #expect(next.halfmoveClock == 0)
}

@Test
func drawDetectionMatchesTypeScriptRules() {
    let quiet = makePosition(side: .playerOne, halfmoveClock: 32)
    let allKings = makePosition(side: .playerOne, p1Kings: B1(10), p2Kings: B1(21), halfmoveClock: 16)
    let active = makePosition(side: .playerOne, p1Men: B1(10), p2Men: B1(21), halfmoveClock: 15)

    #expect(isDrawByInactivity(quiet))
    #expect(isDrawByInactivity(allKings))
    #expect(isDrawByInactivity(active) == false)
}
