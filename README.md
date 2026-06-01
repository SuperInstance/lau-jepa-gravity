# lau-jepa-gravity

> A single `f64` per room that algorithmically adjusts model parameters — like Mandelbrot zoom: a simple number, explored at increasing resolution, revealing structure

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## What This Does

This crate implements a **room-based gravity system** for adaptive model parameter selection. Each room (conversation context) carries a single gravity value in [−1, 1] that captures "what shape of response works here." From that one number, the system deterministically derives full model parameters: temperature, prompt style, token budget, top-p, frequency/presence penalties, and more.

The gravity value is updated from interaction signals using an exponential moving average. Playful rooms drift positive; precise rooms drift negative. The system detects user communication style (playful, precise, narrative, Socratic, direct, mixed) from text heuristics and uses it to shape gravity over time.

Beyond single rooms, the crate provides a **GravityField** that manages gravity across all rooms, supports clustering, routing, snapshots, and a **MandelbrotZoom** progressive generation system that iteratively refines parameters from coarse to fine.

## Key Idea

Most LLM parameter tuning is manual: write prompts, set temperatures, tweak configs. This crate replaces all of that with a single number. One `f64` — the gravity — maps deterministically to a complete model configuration through piecewise-linear functions. It's like zooming into the Mandelbrot set: the input is trivial, but the function that expands it reveals rich structure.

The core insight: **you don't need fine-tuning, prompt templates, or configuration files**. You need one number that captures the conversational vibe, and a well-designed function that expands it into model parameters.

## Install

```toml
[dependencies]
lau-jepa-gravity = "0.1"
```

### Dependencies

- **serde** 1 (with `derive`) — all types are serializable

Dev dependency: **serde_json** 1 for test roundtrips.

## Quick Start

### Basic Gravity

```rust
use lau_jepa_gravity::*;

// Create a gravity value
let mut gravity = Gravity::new();
gravity.update(0.8);   // playful interaction
gravity.update(-0.2);  // slightly serious follow-up
assert!(gravity.is_playful()); // still net-positive

// Derive model parameters from the single number
let params = ModelParams::from_gravity(&gravity);
println!("Temperature: {}", params.temperature);
println!("Style: {:?}", params.system_prompt_style);
```

### Room-Based Gravity

```rust
// Gravity field manages multiple rooms
let mut field = GravityField::new();
field.register_room("dev-room");
field.register_room("chat-room");
field.register_room("story-room");

// Feed interaction signals
let precise = GravitySignal::from_text("debug this error precisely");
let playful = GravitySignal::from_text("lol that's hilarious 😂");
let narrative = GravitySignal::from_text("once upon a time, in a faraway land...");

for _ in 0..5 { field.record("dev-room", &precise); }
for _ in 0..5 { field.record("chat-room", &playful); }
for _ in 0..5 { field.record("story-room", &narrative); }

// Query the field
let summary = field.field_summary();
println!("Most serious: {:?}", summary.most_serious);
println!("Most playful: {:?}", summary.most_playful);

// Route a new signal to the best-matching room
let route = field.route_signal(&playful); // → ["chat-room", ...]
```

### Model Parameters

```rust
let params = ModelParams::from_gravity(&Gravity::from_value(-0.8));
// Serious room → Technical prompt style, low temperature
assert_eq!(params.system_prompt_style, PromptStyle::Technical);
assert!(params.temperature < 0.5);

let params = ModelParams::from_gravity(&Gravity::from_value(0.8));
// Playful room → Creative prompt style, high temperature
assert_eq!(params.system_prompt_style, PromptStyle::Creative);
assert!(params.temperature > 0.9);
```

### Mandelbrot Zoom (Progressive Generation)

```rust
// Start simple, refine progressively — like zooming into the Mandelbrot set
let mut zoom = MandelbrotZoom::new("my-room");
let complexity = zoom.measure_complexity(100, 80);
if zoom.should_zoom_in(0.5) {
    let decisions = zoom.zoom_in(); // returns 4 sub-region decisions
}
```

### Phone-a-Friend (Model Routing)

```rust
// Large model for complex rooms, small model for simple rooms
let paf = PhoneAFriend::new("gpt-4", "gpt-3.5");
let model = paf.select_model(room.gravity.value, 0.5);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Gravity` | The core unit: value ∈ [−1,1], confidence ∈ [0,1], sample count |
| `GravitySignal` | An interaction: user style + response success + timestamp + context |
| `UserStyle` | Enum: `Playful`, `Precise`, `Narrative`, `Socratic`, `Direct`, `Mixed` |
| `ModelParams` | Derived parameters: temperature, max_tokens, prompt style, top_p, penalties |
| `PromptStyle` | Enum: `Technical`, `Conversational`, `Creative`, `Socratic`, `Narrative`, `Minimal` |

### Room Management

| Type | Description |
|------|-------------|
| `RoomGravity` | Per-room state: gravity + history + effective params + decay rate |
| `GravityField` | Map of room names → gravity, with clustering, routing, snapshots |
| `GravitySnapshot` | Immutable snapshot of room gravity at a point in time |
| `FieldSummary` | Summary: room count, average gravity, most playful/serious rooms |

### Advanced

| Type | Description |
|------|-------------|
| `MandelbrotZoom` | Progressive generation: zoom into parameter space for finer control |
| `ProgressiveGeneration` | Track model efficiency across iterations |
| `PhoneAFriend` | Route between large/small models based on gravity complexity |

### Key Methods

```rust
// Gravity
gravity.update(signal);              // EMA update
gravity.is_playful();                // value > 0.3
gravity.is_serious();                // value < -0.3
gravity.distance_to(&other);         // |value₁ - value₂|
gravity.nudge_toward(&target, 0.5); // move toward target by fraction

// GravitySignal
GravitySignal::from_text("hello lol"); // auto-detect style from text
signal.style_value();                   // Playful→0.8, Precise→−0.8, etc.

// ModelParams
ModelParams::from_gravity(&gravity); // deterministic derivation
params.merge(&other, 0.5);          // blend two param sets
params.validate();                   // check all values in range

// GravityField
field.register_room("name");
field.record("name", &signal);
field.gravity_of("name");            // → Option<f64>
field.route_signal(&signal);         // rooms ranked by compatibility
field.cluster();                     // group rooms by gravity similarity
field.decay_all();                   // apply time decay to all rooms
field.snapshot();                    // capture immutable state
```

## How It Works

1. **Style detection**: Text is scored on five axes (playful, precise, narrative, socratic, direct) using keyword matching, punctuation patterns, and length heuristics. The highest-scoring axis determines the `UserStyle`.

2. **Gravity update**: Each interaction produces a combined signal = `style_value × response_success`. The gravity value is updated via exponential moving average with adaptive learning rate α = 1/min(sample_count + 1, 10).

3. **Parameter derivation**: The gravity value maps to `ModelParams` through piecewise-linear functions. Five gravity ranges produce different parameter profiles:

   | Gravity Range | Style | Temperature | Behavior |
   |--------------|-------|-------------|----------|
   | < −0.6 | Technical | 0.3 | Precise, cited, formal |
   | −0.6 to −0.3 | Minimal | 0.5 | Brief, factual |
   | −0.3 to 0.3 | Conversational | 0.7 | Balanced, friendly |
   | 0.3 to 0.6 | Narrative | 0.9 | Storytelling, vivid |
   | > 0.6 | Creative | 1.1 | Imaginative, surprising |

4. **Field management**: The `GravityField` maintains a HashMap of rooms. It supports clustering (group rooms by gravity proximity), routing (match incoming signals to compatible rooms), and time decay (gravity drifts toward zero over time).

5. **Progressive generation**: The `MandelbrotZoom` starts with coarse parameters and iteratively refines them, measuring complexity at each level and deciding whether to zoom in further.

## The Math

### Gravity Update (Exponential Moving Average)

```
α = 1 / min(n + 1, 10)
g_new = g_old × (1 − α) + signal_clamped × α
```

The learning rate decreases with more samples but bottoms out at 0.1 to maintain adaptability.

### Style Detection (Heuristic Scoring)

Each input text is scored on five axes using weighted keyword/pattern matching:

```
score_playful = 2.0·(contains "lol"/"haha") + 1.5·(contains "pun"/"joke") + 1.0·(count "!" > 2)
score_precise = 2.0·(contains "exactly"/"precisely") + 1.5·(contains "error"/"debug") + ...
```

The highest-scoring axis wins. Ties produce `Mixed`.

### Gravity → Model Parameters

The mapping from gravity value g to model parameters is a piecewise function with five regions. Within each region, scalar parameters are linearly interpolated:

```
temperature(g) = 0.3  for g < −0.6
               = 0.5  for −0.6 ≤ g < −0.3
               = 0.7  for −0.3 ≤ g ≤ 0.3
               = 0.9  for 0.3 < g ≤ 0.6
               = 1.1  for g > 0.6
```

### Confidence

```
confidence = 1 − 1/(n + 1)
```

Approaches 1.0 as sample count n → ∞. Rooms with confidence < 0.3 and sample_count > 5 are flagged for escalation.

## Tests

79 unit tests covering:
- Gravity creation, update, and state queries (playful/serious/balanced)
- Style detection from text patterns
- Model parameter derivation for all five gravity ranges
- Parameter validation and merging
- Room gravity recording and decay
- Gravity field: routing, clustering, snapshots
- Field summary statistics
- Phone-a-friend model selection
- Mandelbrot zoom progressive generation
- Serde roundtrip for all types
- Full integration workflow: 3 rooms, different styles, routing + summary

Run with: `cargo test`

## License

MIT
