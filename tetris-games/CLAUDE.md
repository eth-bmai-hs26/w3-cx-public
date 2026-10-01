# Pixel Toys: CNN intuition games

Five single-file HTML games plus `index.html`. They replace `../tetris-games/` (FS26), which is kept for reference only. Scope is **CNN mechanisms only**: no validation, test decks or shortcut learning.

| Game | Teaches | Built from the failure of |
|---|---|---|
| `game1-stencil-hunt.html` | filter, convolution, feature map, one filter per pattern | (start) |
| `game2-layers.html` | a second convolution layer over the sheets, activation | one stencil can't see a whole toy |
| `game3-pooling.html` | max pooling: tolerance, shrinking, and the trade-off | Game 2's inspector is too fussy, and its filter is as big as the Robo |
| `game4-let-it-learn.html` | the forward pass (feature map → global max → sigmoid → cross-entropy) and training | hand-designing stencils doesn't scale |
| `game5-why-a-stencil.html` | CNN vs fully connected: weight sharing (move the part) and locality (count the numbers) | both sorters score 12/12 on the pile, so why use a stencil? |

Superseded files are kept on purpose in `../old/` (outside this folder, not linked from `index.html`; their links back to `index.html` no longer resolve there): `game2-build-the-inspector.html` (v1, layout card and stride-1 "nearby"), `game2-build-the-inspector-v2.html` (layers and pooling in one game; source of Games 2 and 3), `game3-let-it-learn.html` (Game 4 before the pipeline screen), and `game4-let-it-learn-before-game5-split.html` (Game 4 with the second-camera comparison, before it moved to Game 5).

Story: the player is the new QC lead at Pixel Toys, and the supervisor is Mara. All five games share the palette, fonts and theme toggle (`localStorage` key `pixeltoys-theme`, wrapped in try/catch). Google Fonts is the only external fetch, with system fallbacks.

## Invariants that make the teaching moments land

Check these again if you change any data.

**Game 1**
- Tray A has exactly two 4s for the T-stencil (stem down).
- In tray B, each T orientation gives exactly one 4, and the "stem left" stencil gives none.
- Stencils only have 0/1 weights; negative weights were left out on purpose.

**Game 2 v1** (superseded, `../old/game2-build-the-inspector.html`)
- `ROBO` is the only way to tile the outline with the four parts.
- Exact matching accepts a moved Robo and rejects the wobbly one.
- "Nearby" (±1) accepts both good trays and rejects the jumbled and headless ones, at every position in the tray.
- "Anywhere" accepts the jumbled tray. That's the too-much-pooling lesson.
- The layout card is a real layer-2 filter: it slides over the four part sheets (4×5 card, so the Robo sheet is 7×6). At each position it writes how many parts are in place, and a 4 means Robo. The reveal's worked example reads this sheet.
- Pooling is stride 1 ("within ±t squares"). Stride-2 grid pooling makes results depend on alignment. The shrinking job (2×2, stride 2) is only illustrated in the reveal, with a 4×4 → 2×2 example.

**Games 2 and 3** (`game2-layers.html`, `game3-pooling.html`; both built from the v2 file, sharing the Robo, the engine and the four trays)
- A literal two-layer CNN: four 3×3 part stencils (conv) → activation, i.e. ReLU(count − 3), so only full matches pass → max pooling → a 3×3×4 Robo filter (conv) → "is there a 4?" (global max).
- Pooling is a 3×3 window jumping 2, with padding 1 (overlapping, as in AlexNet), so the 10×10 sheets become 5×5. Plain 2×2 stride-2 pooling would forgive a one-square slip only when the block edges happened to fall right.
- The 7×7 Robo: head and body are both T, legs are J/L. Every stencil corner is at even coordinates, so each part lands on its own pooled square. The outline has exactly one tiling.
- Game 2 has no pooling: its Robo filter is 5×5, and its Step 3 asks players to predict three trays (another Robo, headless, then jumbled last as the climax). Game 2 ends on a win. The low-leg Robo first appears in Game 3's hook, where it fails: that rejection is the motivation for pooling, so don't move it back into Game 2.
- Game 3's Step 1 is pooling by hand on a 6×6 sheet with marks at (3,2) and (0,4); the (3,2) mark sits in two overlapping windows. Step 2 is the choice of how much pooling, and the Robo filter's size follows it: none needs 5×5, shrink needs 3×3, squash-to-one-number needs 1×1.
- Checked at every tray position where all stencil windows fit: the good Robo passes in all three modes; the low leg fails with no pooling and passes otherwise; the jumbled tray passes only when squashed to one number; the headless tray always fails.

**Game 4** (`game4-let-it-learn.html`)
- One 3×3 filter plus a bias, no ReLU and no output layer: T-score = sigmoid(best), where best is the global max of the feature map. Trained by full-batch gradient descent on binary cross-entropy (the gradient through the sigmoid is p − y). There are 10 learned numbers: 9 weights plus the bias b, and the bias sets the bar.
- An earlier version had an extra output layer (z = w × best + c). With one filter that layer only rescales, so it was removed: it looked arbitrary to players.
- "Inside one guess" comes before "Think first" and shows the forward pass for one tray at a time (slider, dots, ← →, swipe): feature map (10×10) → global max → sigmoid → cross-entropy, then pile loss and accuracy.
- `SEED = 36`, `LR = 1.5`. It reaches 12/12 in 11 rounds, with accuracy 6,6,6,6,11,10,11,…,12 (one dip at round 6). Loss goes from 0.75 to 0.42. Training stops at 12/12 so the "what it learned" text stays true.
- Learned weights ≈ `[[.57,-.05,.74],[-.85,1.0,-.86],[-.12,-.18,-.05]]`, b ≈ −1.28: blue on the bar ends and the stem, red beside the stem. The "what it learned" text depends on this shape. Most other seeds converge slowly or not at all.
- `PAD = 2` (the stencil may hang off the edge). This makes the CNN exactly shift-equivariant.

**Game 5** (`game5-why-a-stencil.html`; same engine, seed and pile as Game 4)
- It opens on the training pile, with an 8×8 "where plastic appeared" count map: every part sits in the top-left 4×4 corner. Then both sorters train live, side by side, from random starts (stencil seed 36, fully connected `SEED_PX = 7`). The stencil sorter is done after 11 rounds, the fully connected one after 23. Its numbers outside the outlined corner never move from their random start. Jumping ahead with the dots trains both instantly (`ensureBoth`).
- Idea 1, move the part: the stencil sorter gives the same T-score (0.73) at every position. It finds every T and never calls another part a T, anywhere. The fully connected sorter (64 weights + bias) only learned the top-left 4×4 corner. In the bottom-right its weights are still random, so its answer is luck: with seed 7 it finds 2 of the 6 bottom-right T positions, says 0.49 ("not a T") at the corner spot (6,5), and its score jumps around (0.44–0.57) as the part moves.
- Idea 2, count the numbers: 10 against n² + 1 for picture sizes 8, 28, 224 and 1,000. Locality is explained here, as the reason the stencil stays small.
- The piles are balanced 6/6. With an unbalanced pile, both learners just predict "not T".
