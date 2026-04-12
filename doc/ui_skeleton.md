# UI Skeleton Spec

Skeleton layout spec for the `uiNew` branch.

Goal:
- lock structure before final art
- define spacing, card sizes, image placeholders, CTA sizes
- support a free-to-play model with ads and a paid "remove ads" option

Reference device:
- width: `390`
- height: `844`
- safe area top: `16`
- safe area bottom: `16`

Responsive rule:
- keep all horizontal measurements proportional between `360` and `430`
- board screens should prioritize keeping the board square fully visible

---

## Design Tokens

### Spacing

| Token | Value |
|---|---|
| `space-4` | `4` |
| `space-8` | `8` |
| `space-12` | `12` |
| `space-16` | `16` |
| `space-20` | `20` |
| `space-24` | `24` |
| `space-32` | `32` |

### Radius

| Token | Value |
|---|---|
| `radius-12` | `12` |
| `radius-16` | `16` |
| `radius-20` | `20` |
| `radius-24` | `24` |

### Core Components

| Component | Size |
|---|---|
| Primary CTA height | `56` |
| Secondary CTA height | `48` |
| Toggle height | `56` |
| Difficulty card height | `92` |
| Small icon button | `40 x 40` |
| Large thumbnail | `64 x 64` |
| Hero art placeholder | `88 x 88` |

### Main Content Frame

| Item | Value |
|---|---|
| Page horizontal padding | `20` |
| Max content width | `350` |
| Section gap | `24` |
| Card internal padding | `14` |

---

## Difficulty Ladder

Use 5 levels:

1. Beginner
2. Easy
3. Medium
4. Hard
5. Master

Recommended internal mapping:

| UI Level | Engine Mapping |
|---|---|
| Beginner | lighter than current `easy` |
| Easy | current `easy` |
| Medium | current `medium` |
| Hard | current `hard` |
| Master | current `hard` plus stronger budget later |

---

## Screen 1: Home

Target file:
- `src/ui/HomeScreen.tsx`

Purpose:
- first landing screen
- entry point to play, settings, shop, and arena/debug later

### Layout

Top to bottom:

1. Hero block
2. Mode block
3. Difficulty preview block
4. Primary CTA
5. Secondary actions

### Measurements

| Element | Spec |
|---|---|
| Page padding left/right | `20` |
| Hero top margin from safe area | `20` |
| Hero art | `88 x 88` |
| Gap: hero art to title | `16` |
| Title area height | `52` |
| Subtitle area height | `24` |
| Mode toggle container | `56` high |
| Gap between major sections | `24` |
| Primary CTA | `56` high |
| Secondary action row | `48` high buttons |

### Suggested structure

```text
[safe area]

      [hero art 88x88]
      [game title]
      [subtitle]

[mode label]
[ vs AI ] [ vs Human ]

[difficulty label]
[selected difficulty preview card]

[ Start Game ]

[ Shop ] [ Settings ]
[ Arena / Labs ]   optional for non-player build
```

### Difficulty preview card

| Element | Spec |
|---|---|
| Card height | `92` |
| Card radius | `20` |
| Card padding | `14` |
| Thumbnail placeholder | `64 x 64` |
| Gap thumbnail to text | `12` |
| Right-side arrow / badge | `24 x 24` |

---

## Screen 2: Difficulty Select

Purpose:
- dedicated screen for choosing one of 5 difficulty levels
- each level gets a fixed artwork slot

### Layout

| Element | Spec |
|---|---|
| Header row height | `44` |
| Header bottom gap | `20` |
| Difficulty list item height | `92` |
| Gap between items | `12` |
| Sticky bottom CTA area | `88` |

### Card structure

Each difficulty card:

| Element | Spec |
|---|---|
| Card outer height | `92` |
| Internal padding | `14` |
| Thumbnail frame | `64 x 64` |
| Title height | `22` |
| Subtitle height | `18` |
| Meta line height | `16` |
| Selection ring / badge | `28 x 28` |

### List wireframe

```text
[Back]             [Difficulty]
[choose your challenge]

[img] Beginner  very relaxed play          [ ]
[img] Easy      casual game                [ ]
[img] Medium    balanced default           [x]
[img] Hard      stronger tactics           [ ]
[img] Master    strongest mobile mode      [ ]

[ Continue ]
```

### Artwork handoff rule

For all 5 cards:
- art safe box: `56 x 56`
- frame box: `64 x 64`
- optional glow/background shape may extend to `72 x 72`

---

## Screen 3: Side Select

Purpose:
- choose to go first or second against AI

### Layout

| Element | Spec |
|---|---|
| Top illustration | `120 x 120` |
| Side card height | `96` |
| Gap between side cards | `12` |
| CTA bottom area | `88` |

### Wireframe

```text
[top illustration]
[Choose your side]
[First move or second move]

[P1 card - go first]
[P2 card - go second]

[ Start Match ]
```

---

## Screen 4: Match Screen

Target file:
- `src/ui/HumanVsCodexArenaScreen.tsx`

Purpose:
- actual gameplay
- must stay clean even if monetized

Important rule:
- do not put a persistent banner ad inside the board area

### Layout priority

1. top status
2. board
3. move helper / action strip
4. bottom utility buttons

### Measurements

| Element | Spec |
|---|---|
| Horizontal padding | `16` |
| Header height | `72` |
| Board frame width | `min(screenWidth - 32, 358)` |
| Board frame height | same as width |
| Gap header to board | `12` |
| Action strip height | `52` |
| Bottom controls row | `56` |

### Board zone

| Element | Spec |
|---|---|
| Board card padding | `8` |
| Board outer radius | `24` |
| Captured/status side chips | `32` high |
| Turn badge | `36` high |

### Match wireframe

```text
[back] [player names / difficulty] [menu]
[turn status / AI thinking]

[ board square area ]

[hint / move info / engine status]

[ New Game ] [ Pause ]
```

### Ad placement guidance

Recommended:
- interstitial only after result screen
- rewarded ad from result screen or continue flow

Avoid:
- forced interruption in the middle of a match
- banner under the board if it compresses touch targets

---

## Screen 5: Result Screen

Purpose:
- win / lose / draw feedback
- main monetization checkpoint for free users

### Layout

| Element | Spec |
|---|---|
| Result hero icon | `96 x 96` |
| Result card padding | `20` |
| Result card radius | `24` |
| CTA stack gap | `12` |
| Reward card height | `84` |

### Wireframe

```text
[hero icon]
[You Win / You Lose / Draw]
[short summary]

[ Rematch ]
[ Back to Home ]

[ Watch ad for bonus ] optional
[ Remove ads forever ] optional upsell
```

### Monetization placement

Best place for upsell:
- after 1 to 3 finished matches
- on result screen, never before first move

---

## Screen 6: Shop / No Ads

Purpose:
- clean place to explain monetization
- users should understand what paid mode removes

### Layout

| Element | Spec |
|---|---|
| Header row | `44` |
| Product hero card | `160` |
| Benefit rows | `56` each |
| Purchase CTA | `56` |
| Restore button | `48` |

### Wireframe

```text
[Back]               [No Ads]

[ large product card ]
[ remove forced ads ]
[ keep optional rewarded ads ]
[ support the game ]

[ Buy once ]
[ Restore Purchase ]
[ Terms ] [ Privacy ]
```

### Copy guidance

Be explicit:
- "Removes forced ads between matches"
- "Optional rewarded ads may still be available for bonuses"

Do not imply:
- online rank advantage
- stronger AI for paid users

---

## Screen 7: Settings

Purpose:
- lightweight utility screen
- also important for compliance

### Layout

| Element | Spec |
|---|---|
| List row height | `56` |
| Group spacing | `24` |
| Footer legal links area | `72` |

### Sections

1. Audio
2. Gameplay
3. Language
4. Purchases
5. Legal

### Recommended rows

- Sound
- Vibration
- Animation Speed
- Show Move Hints
- Restore Purchase
- Privacy Policy
- Terms of Service

---

## Ad and Purchase Flow

Recommended lightweight flow:

1. User completes match
2. Result screen appears immediately
3. Free user may see interstitial after a small delay or after pressing next
4. After repeated sessions, show a soft `Remove Ads` offer on result screen
5. Settings and Shop always contain `Restore Purchase`

Rules:
- never show forced ad on app launch
- never show forced ad during active gameplay
- first-time user should finish at least one match before heavy monetization

---

## Build Order for `uiNew`

Recommended implementation order:

1. Home skeleton
2. Difficulty select skeleton
3. Side select skeleton
4. Match screen cleanup
5. Result screen
6. Shop / No Ads
7. Settings

---

## Notes for Art Production

When preparing images, use these default slots:

| Asset | Slot |
|---|---|
| Home hero mascot | `88 x 88` |
| Difficulty icons | `64 x 64` |
| Side select illustration | `120 x 120` |
| Result icon | `96 x 96` |
| Shop product emblem | `96 x 96` inside a larger hero card |

If you want, the next step can be:
- convert this spec into actual placeholder React Native screens
- or turn just `Home + Difficulty` into a coded skeleton first
