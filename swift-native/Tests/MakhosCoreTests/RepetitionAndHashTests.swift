import MakhosCore
import Testing

@Test
func repetitionCountersTrackPushAndPop() {
    let hash: UInt32 = 42
    var counts = buildRepetitionCounts([hash, hash])

    #expect(getRepetitionCount(counts, hash: hash) == 2)
    #expect(isThreefoldRepetition(counts, hash: hash) == false)

    let next = pushRepetition(&counts, hash: hash)
    #expect(next == 3)
    #expect(isThreefoldRepetition(counts, hash: hash))

    popRepetition(&counts, hash: hash)
    #expect(getRepetitionCount(counts, hash: hash) == 2)
}

@Test
func zobristHashesAreStableAndReactToMoves() {
    let position = initialPosition()
    let original = hashPosition(position)
    let originalVerify = verifyHashPosition(position)

    let moved = applyMove(position, generateMoves(position)[0])
    let movedHash = hashPosition(moved)
    let movedVerify = verifyHashPosition(moved)

    #expect(original == hashPosition(position))
    #expect(originalVerify == verifyHashPosition(position))
    #expect(original != movedHash)
    #expect(originalVerify != movedVerify)
}
