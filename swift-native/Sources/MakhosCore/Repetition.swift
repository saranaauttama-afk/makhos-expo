public struct RepetitionCounts: Equatable, Sendable {
    public var counts: [UInt32: Int]

    public init(counts: [UInt32: Int] = [:]) {
        self.counts = counts
    }
}

public func buildRepetitionCounts(_ hashes: [UInt32]) -> RepetitionCounts {
    var counts: [UInt32: Int] = [:]
    for hash in hashes {
        counts[hash, default: 0] += 1
    }
    return RepetitionCounts(counts: counts)
}

public func getRepetitionCount(_ counts: RepetitionCounts, hash: UInt32) -> Int {
    counts.counts[hash, default: 0]
}

@discardableResult
public func pushRepetition(_ counts: inout RepetitionCounts, hash: UInt32) -> Int {
    let next = counts.counts[hash, default: 0] + 1
    counts.counts[hash] = next
    return next
}

public func popRepetition(_ counts: inout RepetitionCounts, hash: UInt32) {
    let previous = counts.counts[hash, default: 0]
    if previous <= 1 {
        counts.counts.removeValue(forKey: hash)
    } else {
        counts.counts[hash] = previous - 1
    }
}

public func isThreefoldRepetition(_ counts: RepetitionCounts, hash: UInt32) -> Bool {
    getRepetitionCount(counts, hash: hash) >= 3
}
