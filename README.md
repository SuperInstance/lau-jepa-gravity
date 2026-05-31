# lau-jepa-gravity

A single `f64` per room that algorithmically adjusts model parameters. Think Mandelbrot zoom: a simple number, explored at increasing resolution, revealing structure.

## The concept in 60 seconds

Every room has a **gravity** value — a single `f64` in `[-1, 1]` that captures "what shape of response works here." Playful rooms have positive gravity. Serious rooms have negative gravity. The gravity is updated from interaction signals using exponential moving average.

But here's the trick: from that single number, the system derives full algorithmic model parameters — prompt style, temperature, verbosity, creativity, and more. It's like zooming into the Mandelbrot set: a simple input reveals infinite structure through deterministic functions.

```rust
let gravity = Gravity::from_value(0.6); // playful room
let params = ModelParams::from_gravity(&gravity);
// params.prompt_style → Socratic
// params.temperature  → 0.85
// params.verbosity    → Moderate
```

## Quick start

```rust
use lau_jepa_gravity::*;

// Create a gravity value for a room
let mut gravity = Gravity::new();

// Update it from interaction signals
gravity.update(0.8);  // playful interaction
gravity.update(-0.2); // slightly serious follow-up
assert!(gravity.is_playful()); // still net-positive

// Derive model parameters
let params = ModelParams::from_gravity(&gravity);
println!("Temperature: {}", params.temperature);
println!("Style: {}", params.prompt_style);
println!("Verbosity: {:?}", params.verbosity);

// Gravity field across all rooms
let mut field = GravityField::new();
field.set_gravity("bridge", Gravity::from_value(0.5));
field.set_gravity("engineering", Gravity::from_value(-0.7));

// Query the field
let summary = field.summary();
let clusters = field.cluster(); // group rooms by gravity similarity
```

## Key types

| Type | What it does |
|------|-------------|
| `Gravity` | The core unit: value, confidence, sample count |
| `GravitySignal` | An interaction that shapes gravity (style, satisfaction, outcome) |
| `ModelParams` | Algorithmically derived parameters: temperature, style, verbosity |
| `RoomGravity` | Named gravity with signal history for a single room |
| `GravityField` | Map of room names → gravity values, with clustering |
| `GravitySnapshot` | Immutable snapshot of the full field |
| `MandelbrotZoom` | Progressive generation: simple → complex from a seed |
| `ProgressiveGeneration` | Iteratively refine parameters from coarse to fine |

## Model parameter derivation

```rust
let params = ModelParams::from_gravity(&gravity);
// Deterministically derives:
// - prompt_style:  Direct ↔ Socratic ↔ Narrative
// - temperature:   0.3 (serious) ↔ 0.9 (playful)
// - verbosity:     Terse ↔ Moderate ↔ Detailed
// - creativity:    0.1 ↔ 0.9
// - structure:     Free ↔ Scaffolded
```

No configuration files. No prompt templates. One number → full model config.

## Mandelbrot zoom — progressive generation

```rust
let zoom = MandelbrotZoom::new(0.42, 0.3); // seed coordinates
let gen = ProgressiveGeneration::new(zoom);

// Each iteration reveals more structure
for i in 0..5 {
    let params = gen.iterate();
    println!("Iteration {}: temp={:.2}", i, params.temperature);
}
```

## Contributing

PRs welcome. This crate is part of the [SuperInstance](https://github.com/SuperInstance) ecosystem. The gravity-to-params mapping is the core insight — if you have ideas for better parameter derivation functions, open an issue.
