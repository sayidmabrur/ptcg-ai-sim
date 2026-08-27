# `simulator/` — play the game yourself, against a trained agent

An interactive web front end on top of the competition engine, which lives here
in full as a single package: `ptcg_engine/src/` is the C++ source and
`ptcg_engine/*.py` the ctypes bindings that call into it. `./build_engine.sh`
compiles the former into the `libcg.so` the latter loads — so the engine the
simulator runs is the engine in this repository, not a binary of unknown
vintage. The build also installs the same binary into every agent bundle's own
`cg/`, because a bundle runs in its own interpreter and must not depend on this
one. Rebuild after any engine update, and once after cloning: no `.so` is kept
in git, so building *is* how the engine arrives.

    ./build_engine.sh

The build needs a C++20 compiler and takes about 30 seconds. It passes
`-include climits` because the source reaches for `INT_MAX` without including
it — MSVC supplies that transitively, GCC and Clang do not — which is a compiler
flag rather than an edit, since the package is competition-licensed and should
stay untouched. Verified against the shipped binary: both expose the same
thirteen entry points and produce byte-identical card and attack tables.

Nothing in this directory reaches outside it — the training folders can be moved
away or left out of a checkout and the simulator still runs.

You pilot one seat in a browser; a registered agent from `agents/` pilots the
other. No game rule is implemented here: every legal move, every coin flip and
every card effect comes from the engine, and the UI only ever offers the exact
option list the engine put in front of the seat you are sitting in.

## Run it

    ./build_engine.sh          # once per checkout — no .so is kept in git, so
                               # building is how the engine arrives
    PY=/home/capon/miniconda3/envs/kaggle-pokemon/bin/python
    $PY simulator/server.py --port 8000
    # then open http://127.0.0.1:8000

Card art lives in `simulator/card_images_en/` — 683 MB, kept out of git. The
lookup falls back to `../card_images_en`, and `PTCG_CARD_IMAGES=/path/to/scans`
overrides both. Without any of them the board draws text cards instead and
everything else works unchanged, so a fresh clone is playable before the scans
arrive.

**New game** opens a dialog rather than a row of dropdowns, because a decklist
is 60 cards and the only useful way to show one is to show the cards. The four
prebuilt lists — Crustle, Grimmsnarl ex, Alakazam, Mega Lucario ex — each render
as one tile per distinct card with its count in the corner (`[image] x4`), so a
list reads at a glance. Pick a deck, an opponent and a seat, then **Start game**.

### Choosing an opponent

The opponent is not a dropdown entry but a card of its own: the archetype's
headline Pokémon (picked from the agent's decklist — the most numerous Pokémon
ex in it), the architecture that pilots it, and the 60 cards it plays.

The architecture is read from the checkpoint rather than described by hand.
`describe_agent.py` walks the state dict — tensor shapes give the width, layer
indices give the depth of each tower, the total gives the parameter count — and
takes the head count and feed-forward multiplier from the constructor defaults
in the bundle's own policy module, parsed with `ast` so that describing one
bundle cannot import a module name a second bundle also uses. The result lands
in `ARCH.json` beside the weights:

    python describe_agent.py agents/grimmsnarl_ex_xattn

    Cross-attention pointer transformer
    Parameters        2,685,443       Decision chain     30 frames
    Embedding dim     64              Opponent history   30 frames
    Attention heads   2               Trained by         behavioural cloning
    Layers            20 across 4 towers (chain 8, context 2, history 8, cross 2)

Run it once when registering a bundle; the server reads the JSON, so it never
imports torch itself.

### Create your own…

The fifth entry swaps the preview for a builder over the whole 1267-card pool:

- **Search** by name, ability text or attack text, with a card-type filter.
- **Add to Deck** / **Add x4** for the selected card; double-click a result to
  add one; hover +/− on a deck tile to adjust a count; **Clear** to start over.
- **Start from a prebuilt deck…** copies a legal list in as a base — the fastest
  way to build is to edit something that already works.
- **Export CSV** copies the deck as one card id per line (the same `deck.csv`
  format the bundles ship); **Import CSV** pastes one back.

The legality line updates on every change and states what is still wrong in the
engine's own terms — `54/60`, `5 copies of Munkidori (max 4)`, `2 ACE SPEC cards
(max 1)`, `no Basic Pokémon`. Those are the rules `ApiBattleStart` enforces
(`engine.deck_check` mirrors them), so a deck the dialog calls legal is one the
engine will accept; `/api/new` re-checks server-side and refuses anything else.
Deep link `#build` opens straight into the builder.

## The opponent is a policy *and* its deck

There is no separate "agent deck" choice. A checkpoint was trained piloting the
60 cards packaged next to it, and handing it a different list measures neither
the policy nor the deck — so one dropdown entry names both.

Opponents are **registered**: one directory each under `simulator/agents/`,
holding the full inference code the competition would run.

    simulator/agents/grimmsnarl_ex_xattn/
        main.py          def agent(obs_dict) -> list[int]   <- all the simulator calls
        deck.csv         the 60 card ids it was trained on
        actor_critic.pt  the weights
        policy_network/  the generation of the policy modules those weights fit
        cg/              the bundle's own engine bindings

To add an opponent, drop another bundle directory in; the dropdown is built from
whatever is there (`opponents.discover_bundles`). Nothing else is scanned. The
simulator knows only `main.py` and its `agent()` function, so any agent that
answers a competition observation can be registered without the simulator
learning anything about it.

Bundles elsewhere in the repo are deliberately not discovered any more. Several
of them fail to load or never answer, and one of those was what left **New game**
stuck on "starting…" — the server was blocked forever on a worker that would
never reply. Both waits are now bounded (`START_TIMEOUT`, `MOVE_TIMEOUT`), the
worker sends a `ready` handshake once its policy is loaded, and a bundle that
cannot load fails the *start* with its own error message instead of hanging.

Each bundle still runs in its own process (`agent_worker.py`, one JSON
observation per line in, one selection per line out): bundles import their policy
modules by bare name and ship their own `cg`/`libcg.so`, so whichever loaded
first in a shared interpreter would capture those names and run the next
bundle's weights through the wrong architecture. If a bundle raises mid-game the
worker answers with a legal random move — a weak move beats ending your game —
and the lapse reaches the browser console as `agentError`.

## The board

The page is laid out as the physical playmat, not as two lists of cards: the
two Active Spots face each other across the centre line, each player's five
Bench slots sit behind their Active, Prizes stack on the outer left and the deck
and discard pile on the outer right, with your hand along the bottom and the
Stadium on the centre line. The opponent's half is the same grid mirrored.
Empty slots are drawn rather than omitted, so a bench of two reads as two of
five, and a taken prize leaves a gap in the 2x3 block.

Cards are the real scans from `card_images_en/`, one file per card id, served
by the app at `/cards/`. The picture is the card; overlaid on it is only what
the print cannot know — current HP and the damage taken, attached Energy as
pips, tools, Special Conditions, and a "new" marker on a Pokémon played this
turn. Hovering any face-up card enlarges the art next to its full rules text,
attacks and costs. A card with no file on disk falls back to the drawn text
face, which is also what face-down backs use, so the board still works if the
directory is absent.

The opponent's hand is a row of face-down backs, one per card, the way it looks
across a table — counting them is the same information the number was, in the
form the game gives it.

What the mat alone cannot show sits in the nameplates: the four once-per-turn
rights (energy attach, supporter, stadium, retreat) are struck through as this
turn spends them, next to hand and deck counts and any Special Conditions.

Two prompts in a row can look identical — Hilda searches the deck twice, once
for an Evolution Pokémon and once for an Energy card, under the same title —
and both allow taking nothing. A second click meant for the first prompt used to
land on the second and silently answer it "nothing", so Hilda fetched only half
of what it should. Clicks are now ignored for 400 ms after a selection is sent,
and the button reads **Take nothing** rather than an enabled `Confirm (0/1)`, so
skipping a search is something you choose rather than something you fall into.

**Confirm** is wired in `renderChoice`, next to the state it sends, rather than
once at startup — a startup wiring line was lost in an edit and left every
multi-select confirm (Ultra Ball's discard, a bench search) enabled but dead,
which no amount of checking *whether the button was reachable* could catch. The
regression check is now `full_game.py`-style: play a whole game through the UI
and assert it reaches a result.

The choice panel and the log split the column on fixed terms: the log takes a
constant slice of the viewport (`clamp(150px, 28vh, 320px)`) and scrolls inside
it, and the choice panel keeps the rest with a 220px floor. Letting the log size
itself meant it grew all game and squeezed the option list — down to 11 pixels by
turn four — which is what made **End turn** look missing when there was nothing
left to do. (It is always offered: 103 of 103 Main decisions across two games
carried an End-turn option.) With 400 log lines, twice what the server ever
sends, the option box measures the same as it does on turn one.

The panel is also pinned to the viewport and never taller than it — the option
list and the log scroll inside their own boxes — so **Confirm** is always on
screen. It used to sit below the fold on a short window, which reads as a button
that cannot be clicked. Clicking a board slot that several options share now
takes the next one not yet picked (and gives one back when they are all taken),
so a slot click always moves a multi-select towards the count Confirm needs. And
when the engine refuses a selection that passed the count check,
`engine.raw_select` keeps its error code (`cg.game.battle_select` collapses them
all into an empty `IndexError`) and the reason appears next to the prompt with
the picks intact, instead of as an empty message in the corner.

Every legal choice is both a row in the side panel and a highlight on the board:
the slots an option touches are outlined, hovering one highlights the matching
row, and clicking the slot takes the option when only one lives there. Options
that name two places at once -- attaching or evolving, where a card leaves your
hand and lands on a Pokémon -- light up both. Option text is resolved the same
way, so a bench target reads "Attach Basic {D} Energy to Munkidori (bench)"
rather than "a card". Discard piles are clickable, and hovering any face-up card
shows its full rules text, attacks and costs.

## Watching what happened

A turn the opponent took is otherwise invisible: by the time the board comes
back, the Poké Pad has been played, the attack has resolved, and all you have is
a changed number. So each view carries the events behind its log lines
(`render.log_events`) and the board acts them out in order:

- a card played is held up in a banner with its art — the agent's whole turn
  replays as a sequence of these, which is what makes its decisions legible;
- an attack announces itself (name and damage) while the attacking Pokémon
  lunges;
- HP changes float off the Pokémon they happened to — red `-120`, green `+80` —
  and the card flashes as it takes them;
- evolutions pop, attachments flash, coin flips and Special Conditions get a
  banner.

**Check which build you are running.** The header shows `build <n>` and the
console prints the same on load. If that number does not change after you
restart the server, the page is running from cache and nothing described here is
loaded. The asset URLs are versioned by file mtime (`/static/app.js?v=…`) so
that cannot happen any more: a changed file gets a changed URL, which no cache
can answer. `no-cache` headers alone were not enough — a browser that cached
`app.js` *before* those headers existed kept serving its old copy without
asking, which is how a paced animation could still look instant on your screen
while measuring correctly on mine.

**Your options are locked while the opponent plays.** A turn arrives as a batch
of events, and acting immediately used to cancel the rest of the replay — the
very thing it exists to show, and easy to do by accident. The panel reads
*Opponent is playing…* until the replay ends.

There is deliberately nothing on screen to skip with. A button in the panel and
a click-anywhere-over-the-board shortcut both existed and both were too easy to
hit by accident, and losing the turn you were waiting to watch is the one thing
this must not do. The **S** key still cuts a replay short — a key is not
something you press by mistake while watching.

**The board moves with the narration, one action at a time.** This is the part
that took three attempts to get right. The pop-ups were paced from the start,
but the board was drawn from the *end-of-turn* position the moment the response
arrived — so the banner said "played Poké Pad" over cards that had already made
every move of the turn. Pacing an overlay over a board that has finished moving
is not an animation.

`engine.Battle.advance` now takes a board snapshot after **each** of the agent's
decisions (`render.board_snapshot`, hands stripped — the agent's is private and
the human's is not in that observation at all), and each snapshot is tagged with
its position in the log. The browser rewinds to the board as it stood before the
agent moved and walks it forward as the narration plays, so the cards arrive one
by one with the sentence describing them, and the real position is drawn at the
end. Measured from the DOM:

    +31497 ms  SAY    Opponent · Card moved · hand → bench
    +31497 ms  BOARD  opp active=Snorunt bench=[Munkidori] hand=6
    +32602 ms  SAY    Opponent · Draw · drew a card
    +32603 ms  BOARD  opp ... hand=5
    +33604 ms  SAY    Opponent · Spikemuth Gym · played

**Cards travel between zones.** A snapshot swap alone means a drawn card simply
appears in hand and a Knocked Out Pokémon simply stops existing, so each move
also sends a copy of the card flying from where it was to where it is going —
deck to hand on a draw (fanned, up to four, when several are drawn at once),
board to discard on a Knock Out, shrinking and fading on the way. The travelling
card is face-up only when the move is public and the card is known: a card going
into the opponent's hand flies face-down, because it is not ours to show. Only
the engine's own movement entries are animated, never the "played" line as well,
so a card is never sent twice or to the wrong pile.

**Every action gets at least a second.** The opponent's whole turn arrives in one
response, so this pacing is the only thing between you and a board that changes
all at once: `MIN_BEAT` is 1 s at normal speed, and the actions worth dwelling on
get more (an attack 1.8 s, a played card 1.4 s). There is no length-based
compression — a turn with twelve actions in it takes twelve beats to watch,
which is the point. The replay covers the whole turn, not just its highlights:
draws, deck searches, shuffles, switches, mulligans and card movements are shown
alongside plays and attacks, which is what makes a Supporter like Lillie's
Determination or Unfair Stamp *visible* rather than a board that silently
changed. Runs of bookkeeping are merged so this stays watchable — six draws
become one "drew 6 cards" beat, and three 10-damage counters on one Pokémon
become a single -30 — which takes a 37-event turn down to 8 beats. **Replay speed** in the header (slow / normal / fast,
remembered in `localStorage`) is how you shorten it, and the status bar counts
the steps: `replaying the opponent's turn — 3/9`.

Events name their card by `serial`, which is what the rendered board carries, so
an animation anchors to the right slot without any position being sent — and a
card that has already left play simply floats its number from the centre. A
long agent turn plays at double speed, and a newer view cancels whatever is
still queued. All of it is overlay: the board underneath is already the current
position, so an interrupted or dropped animation can never leave it wrong.

One gap is honest rather than papered over: **the engine emits no log entry for
an Ability**, so an opponent's ability use is not in the observation and nothing
is shown for it. Your own is announced from your click, which is the only place
that information exists.

A Pokémon in play can be *face-down*, which the engine reports as a slot that
exists but holds nothing identifiable (`active == [None]`, as against `[]` for a
genuinely empty Active Spot). That is where both players stand through Set Up,
and where the opponent stays until the reveal. The board draws a card back for
it — collapsing the two states made Set Up look like an empty mat and the
Pokémon appear from nowhere on the first turn. Your own Active is face-down to
you as well during Set Up: the engine does not put its identity in your
observation, exactly as the card lies face-down on the table.

## Imperfect information

The engine emits observation JSON for whichever seat owns the current decision,
with the other seat's hand, deck and prize identities erased
(`State::erasePlayerData`, `ToJson.h`). This server keeps **only the human
seat's** observations (`engine.Battle.view_obs`) and throws the agent's away, so
the payload the browser receives has no opponent hand, no deck order and no
prize contents in it — there is nothing hidden client-side that a devtools
window could uncover. Your own prizes are face-down to you too, exactly as in
the physical game. While the agent is choosing, the board you see is the last
position you legitimately observed.

The one place this needs care is a match that *ends* on the agent's turn: that
result lives in the agent's final observation. `engine.public_logs` takes only
the entries attributed to you plus the `RESULT` entry from it, and drops the
rest.

## Files

| file | what it does |
| --- | --- |
| `server.py` | FastAPI app: `/api/config`, `/api/decks`, `/api/cards`, `/api/deck/check`, `/api/new`, `/api/state`, `/api/select`, plus `/cards/` art |
| `engine.py` | one live battle, driven seat by seat; deck loading; log filtering |
| `render.py` | raw observation → the view model the browser draws (card names, option sentences, log lines) |
| `opponents.py` | the registry under `agents/`, and the pipe to a bundle's process |
| `agents/` | one directory per registered opponent (`main.py`, `deck.csv`, weights, `ARCH.json`) |
| `describe_agent.py` | reads a checkpoint and writes the `ARCH.json` the picker shows |
| `agent_worker.py` | one bundle, loaded in its own interpreter, answering selections |
| `static/` | the page (`index.html`, `app.js`, `style.css`), served `no-cache` so an edited asset survives a plain reload |
| `ptcg_engine/` | the whole engine: `src/` (C++) and the ctypes bindings; `build_engine.sh` makes `libcg.so` |
| `decks/` | the four prebuilt decklists, one card id per line |
| `card_images_en/` | card scans, kept out of git (683 MB) — see above |

`render.py` began as the draft in `training_PPO_grimsnarl_ex/webplay/`; its log
renderer was rewritten here against the real `LogType` fields (`value`, `head`,
`isRecover`, the `*_REVERSE` variants).

## One game per process

`cg.sim.Battle` keeps the native battle pointer on a class attribute, so a
process holds exactly one battle. The server is single-game and serialises
requests behind a lock; starting a new game finishes the previous one. Run a
second process on another port if you want two games at once.
