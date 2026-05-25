#if canImport(SwiftUI)
import MakhosCore
import SwiftUI

@MainActor
public final class ArenaViewModel: ObservableObject {
    @Published public private(set) var position: Position
    @Published public private(set) var selectedFrom: Int?
    @Published public private(set) var legalMoves: [Move]
    @Published public private(set) var lastMove: Move?
    public var config: GameConfig

    public init(config: GameConfig = .init(), position: Position = initialPosition()) {
        self.config = config
        self.position = position
        self.selectedFrom = nil
        self.legalMoves = generateMoves(position)
        self.lastMove = nil
    }

    public var fromSquares: [Int] {
        Array(Set(legalMoves.map(\.from))).sorted()
    }

    public var destinationSquares: [Int] {
        guard let selectedFrom else { return [] }
        return legalMoves.filter { $0.from == selectedFrom }.map(\.to).sorted()
    }

    public var statusText: String {
        if config.mode == .vsAI {
            return "Swift native rewrite foundation: board rules are playable now, AI port is next."
        }
        return "Swift native board foundation"
    }

    public func reset() {
        position = initialPosition()
        selectedFrom = nil
        legalMoves = generateMoves(position)
        lastMove = nil
    }

    public func tapSquare(_ square: Int) {
        let movesFromSquare = legalMoves.filter { $0.from == square }
        if !movesFromSquare.isEmpty {
            selectedFrom = square
            return
        }

        guard let selectedFrom,
              let move = legalMoves.first(where: { $0.from == selectedFrom && $0.to == square }) else {
            self.selectedFrom = nil
            return
        }

        position = applyMove(position, move)
        legalMoves = generateMoves(position)
        lastMove = move
        self.selectedFrom = nil
    }
}

public struct RootView: View {
    @State private var route: Route = .home
    @State private var config = GameConfig()
    @StateObject private var arena = ArenaViewModel()

    public init() {}

    public var body: some View {
        NavigationStack {
            switch route {
            case .home:
                HomeView(
                    onQuickPlay: {
                        config = GameConfig(mode: .vsHuman, difficulty: .easy, humanSide: 1, unlimitedThink: false)
                        arena.config = config
                        arena.reset()
                        route = .arena
                    },
                    onSetup: { route = .setup },
                    onAccount: { route = .account }
                )
            case .setup:
                SetupView(
                    config: $config,
                    onStart: {
                        arena.config = config
                        arena.reset()
                        route = .arena
                    }
                )
            case .arena:
                ArenaView(viewModel: arena)
            case .account:
                AccountPlaceholderView()
            }
        }
    }

    enum Route {
        case home
        case setup
        case arena
        case account
    }
}

public struct HomeView: View {
    let onQuickPlay: () -> Void
    let onSetup: () -> Void
    let onAccount: () -> Void

    public var body: some View {
        VStack(spacing: 16) {
            Spacer()
            Text("MAKHOS")
                .font(.largeTitle.bold())
            Button("Quick Play", action: onQuickPlay)
                .buttonStyle(.borderedProminent)
            Button("Setup Match", action: onSetup)
                .buttonStyle(.bordered)
            Button("Account", action: onAccount)
                .buttonStyle(.bordered)
            Spacer()
        }
        .padding()
        .navigationTitle("Home")
    }
}

public struct SetupView: View {
    @Binding var config: GameConfig
    let onStart: () -> Void

    public var body: some View {
        Form {
            Picker("Mode", selection: $config.mode) {
                Text("vs AI").tag(GameMode.vsAI)
                Text("vs Human").tag(GameMode.vsHuman)
            }
            Picker("Difficulty", selection: $config.difficulty) {
                ForEach(Difficulty.allCases, id: \.self) { difficulty in
                    Text(difficulty.rawValue.capitalized).tag(difficulty)
                }
            }
            Picker("Human Side", selection: $config.humanSide) {
                Text("Player 1").tag(1)
                Text("Player 2").tag(-1)
            }
            Toggle("Unlimited Think", isOn: $config.unlimitedThink)
            Button("Start Swift Match", action: onStart)
        }
        .navigationTitle("Setup")
    }
}

public struct ArenaView: View {
    @ObservedObject var viewModel: ArenaViewModel

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(viewModel.statusText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                BoardView(
                    position: viewModel.position,
                    fromSquares: Set(viewModel.fromSquares),
                    selectedFrom: viewModel.selectedFrom,
                    destinationSquares: Set(viewModel.destinationSquares),
                    lastMove: viewModel.lastMove,
                    onTapSquare: viewModel.tapSquare
                )

                Text("Side to move: \(viewModel.position.side == .playerOne ? "Player 1" : "Player 2")")
                Text("Legal moves: \(viewModel.legalMoves.count)")
                Button("Reset Match", action: viewModel.reset)
            }
            .padding()
        }
        .navigationTitle("Arena")
    }
}

public struct AccountPlaceholderView: View {
    public init() {}

    public var body: some View {
        List {
            Label("Settings, purchases, haptics, and persistence move here in the Swift rewrite.", systemImage: "gear")
            Label("The current milestone focuses on the native board foundation and core rules port.", systemImage: "hammer")
        }
        .navigationTitle("Account")
    }
}

public struct BoardView: View {
    let position: Position
    let fromSquares: Set<Int>
    let selectedFrom: Int?
    let destinationSquares: Set<Int>
    let lastMove: Move?
    let onTapSquare: (Int) -> Void

    public init(
        position: Position,
        fromSquares: Set<Int>,
        selectedFrom: Int?,
        destinationSquares: Set<Int>,
        lastMove: Move?,
        onTapSquare: @escaping (Int) -> Void
    ) {
        self.position = position
        self.fromSquares = fromSquares
        self.selectedFrom = selectedFrom
        self.destinationSquares = destinationSquares
        self.lastMove = lastMove
        self.onTapSquare = onTapSquare
    }

    public var body: some View {
        VStack(spacing: 0) {
            ForEach(0..<8, id: \.self) { row in
                HStack(spacing: 0) {
                    ForEach(0..<8, id: \.self) { column in
                        squareView(row: row, column: column)
                    }
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.black.opacity(0.2), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private func squareView(row: Int, column: Int) -> some View {
        let dark = ((row + column) & 1) == 1
        let square = dark ? toIndex(r: row, c: column) : -1
        let piece = dark ? pieceAt(square: square) : nil
        let isFrom = fromSquares.contains(square)
        let isSelected = selectedFrom == square
        let isDestination = destinationSquares.contains(square)
        let isLast = lastMove?.to == square || lastMove?.from == square

        Button {
            if square >= 0 { onTapSquare(square) }
        } label: {
            ZStack {
                Rectangle()
                    .fill(dark ? Color(red: 0.25, green: 0.46, blue: 0.42) : Color(red: 0.95, green: 0.93, blue: 0.88))

                if isLast {
                    Circle().fill(Color.yellow.opacity(0.25)).padding(6)
                }
                if isFrom {
                    Circle().stroke(Color.red.opacity(0.8), lineWidth: 3).padding(7)
                }
                if isSelected {
                    Circle().stroke(Color.white.opacity(0.95), lineWidth: 3).padding(10)
                }
                if isDestination {
                    Circle().fill(Color.orange.opacity(0.5)).frame(width: 16, height: 16)
                }

                if let piece {
                    Circle()
                        .fill(piece.side == .playerOne ? Color(red: 0.28, green: 0.32, blue: 0.38) : Color(red: 0.89, green: 0.91, blue: 0.95))
                        .padding(8)
                    if piece.isKing {
                        Image(systemName: "crown.fill")
                            .font(.caption)
                            .foregroundStyle(.yellow)
                    }
                } else if dark, square >= 0 {
                    Text("\(square + 1)")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .frame(maxWidth: .infinity)
            .aspectRatio(1, contentMode: .fit)
        }
        .buttonStyle(.plain)
        .disabled(square < 0)
    }

    private func pieceAt(square: Int) -> (side: Side, isKing: Bool)? {
        let mask = B1(square)
        if (position.p1Kings & mask) != 0 { return (.playerOne, true) }
        if (position.p1Men & mask) != 0 { return (.playerOne, false) }
        if (position.p2Kings & mask) != 0 { return (.playerTwo, true) }
        if (position.p2Men & mask) != 0 { return (.playerTwo, false) }
        return nil
    }
}
#endif
