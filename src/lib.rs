use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Core Primitive ───────────────────────────────────────────────────────────

/// THE single number — the gravity value that captures "what shape of response works here"
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Gravity {
    pub value: f64,
    pub confidence: f64,
    pub sample_count: u32,
}

impl Default for Gravity {
    fn default() -> Self {
        Self::new()
    }
}

impl Gravity {
    pub fn new() -> Self {
        Self { value: 0.0, confidence: 0.0, sample_count: 0 }
    }

    pub fn from_value(value: f64) -> Self {
        Self {
            value: value.clamp(-1.0, 1.0),
            confidence: 0.5,
            sample_count: 1,
        }
    }

    /// Update gravity based on an interaction signal using exponential moving average
    pub fn update(&mut self, signal: f64) {
        let alpha = if self.sample_count == 0 { 1.0 } else { 1.0 / (self.sample_count as f64 + 1.0).min(10.0) };
        self.value = self.value * (1.0 - alpha) + signal.clamp(-1.0, 1.0) * alpha;
        self.value = self.value.clamp(-1.0, 1.0);
        self.sample_count += 1;
        // Confidence approaches 1.0 with more samples, decays slightly with each update
        self.confidence = (1.0 - (1.0 / (self.sample_count as f64 + 1.0))).min(1.0);
    }

    pub fn is_playful(&self) -> bool {
        self.value > 0.3
    }

    pub fn is_serious(&self) -> bool {
        self.value < -0.3
    }

    pub fn is_balanced(&self) -> bool {
        self.value.abs() <= 0.3
    }

    pub fn distance_to(&self, other: &Gravity) -> f64 {
        (self.value - other.value).abs()
    }

    /// Move gravity toward a target by a fraction (strength 0.0–1.0)
    pub fn nudge_toward(&mut self, target: &Gravity, strength: f64) {
        let strength = strength.clamp(0.0, 1.0);
        self.value = self.value + (target.value - self.value) * strength;
        self.value = self.value.clamp(-1.0, 1.0);
    }
}

// ─── Gravity Signal ───────────────────────────────────────────────────────────

/// How a user communicates — detected from text heuristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UserStyle {
    Playful,
    Precise,
    Narrative,
    Socratic,
    Direct,
    Mixed,
}

/// An interaction that shapes gravity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitySignal {
    pub user_style: UserStyle,
    pub response_success: f64,
    pub timestamp: u64,
    pub context: String,
}

impl GravitySignal {
    pub fn from_text(text: &str) -> Self {
        let style = Self::detect_style(text);
        Self {
            user_style: style,
            response_success: 0.5,
            timestamp: 0,
            context: text.to_string(),
        }
    }

    pub fn style_value(&self) -> f64 {
        match self.user_style {
            UserStyle::Playful => 0.8,
            UserStyle::Narrative => 0.3,
            UserStyle::Socratic => 0.0,
            UserStyle::Direct => -0.3,
            UserStyle::Precise => -0.8,
            UserStyle::Mixed => 0.0,
        }
    }

    fn detect_style(text: &str) -> UserStyle {
        let lower = text.to_lowercase();
        let len = text.len().max(1);

        // Heuristic scoring
        let playful_score = {
            let mut s = 0.0;
            if lower.contains("lol") || lower.contains("haha") || lower.contains("😂") || lower.contains("😄") { s += 2.0; }
            if lower.contains("pun") || lower.contains("joke") || lower.contains("funny") { s += 1.5; }
            if lower.contains("!") && lower.matches('!').count() > 2 { s += 1.0; }
            s
        };

        let precise_score = {
            let mut s = 0.0;
            if lower.contains("exactly") || lower.contains("precisely") || lower.contains("specifically") { s += 2.0; }
            if lower.contains("error") || lower.contains("bug") || lower.contains("fix") || lower.contains("debug") { s += 1.5; }
            if lower.contains("calculate") || lower.contains("measure") || lower.contains("define") { s += 1.5; }
            // Short, command-like text
            if len < 30 { s += 0.5; }
            s
        };

        let narrative_score = {
            let mut s = 0.0;
            if lower.contains("story") || lower.contains("imagine") || lower.contains("picture this") { s += 2.0; }
            if lower.contains("once upon") || lower.contains("suddenly") || lower.contains("meanwhile") { s += 2.0; }
            if len > 200 { s += 1.0; }
            s
        };

        let socratic_score = {
            let mut s = 0.0;
            let q_count = lower.matches('?').count();
            if q_count >= 3 { s += 2.0; }
            else if q_count >= 1 { s += 1.0; }
            if lower.contains("why") || lower.contains("how") || lower.contains("what if") { s += 1.0; }
            s
        };

        let direct_score = {
            let mut s = 0.0;
            if len < 15 { s += 2.0; }
            if lower.starts_with("do ") || lower.starts_with("get ") || lower.starts_with("show ") || lower.starts_with("tell ") { s += 1.5; }
            if !lower.contains('.') && !lower.contains('?') && !lower.contains('!') && len < 20 { s += 1.0; }
            s
        };

        let scores = [
            (UserStyle::Playful, playful_score),
            (UserStyle::Precise, precise_score),
            (UserStyle::Narrative, narrative_score),
            (UserStyle::Socratic, socratic_score),
            (UserStyle::Direct, direct_score),
        ];

        let max = scores.iter().fold(0.0f64, |acc, &(_, s)| acc.max(s));
        if max == 0.0 {
            return UserStyle::Mixed;
        }

        let winners: Vec<_> = scores.iter().filter(|&&(_, s)| s == max).collect();
        if winners.len() > 1 {
            return UserStyle::Mixed;
        }
        winners[0].0
    }
}

// ─── Prompt Style & Model Params ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PromptStyle {
    Technical,
    Conversational,
    Creative,
    Socratic,
    Narrative,
    Minimal,
}

impl PromptStyle {
    pub fn prompt_text(&self) -> &'static str {
        match self {
            PromptStyle::Technical => "Be precise, cite sources, show work",
            PromptStyle::Conversational => "Be friendly, use examples, be helpful",
            PromptStyle::Creative => "Be imaginative, use metaphors, surprise me",
            PromptStyle::Socratic => "Ask questions, guide discovery, don't give answers",
            PromptStyle::Narrative => "Tell a story, use vivid language, paint pictures",
            PromptStyle::Minimal => "Be brief, just the facts, no filler",
        }
    }
}

/// Algorithmic model parameters derived from gravity — NOT fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelParams {
    pub temperature: f64,
    pub max_tokens: u32,
    pub system_prompt_style: PromptStyle,
    pub top_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub seed_options: Vec<u64>,
}

impl ModelParams {
    pub fn from_gravity(gravity: &Gravity) -> Self {
        let v = gravity.value;
        match v {
            v if v < -0.6 => Self {
                temperature: 0.3,
                max_tokens: 2000,
                system_prompt_style: PromptStyle::Technical,
                top_p: 0.8,
                frequency_penalty: 0.3,
                presence_penalty: 0.3,
                seed_options: vec![42, 137, 256],
            },
            v if v < -0.3 => Self {
                temperature: 0.5,
                max_tokens: 1000,
                system_prompt_style: PromptStyle::Minimal,
                top_p: 0.85,
                frequency_penalty: 0.2,
                presence_penalty: 0.2,
                seed_options: vec![7, 99, 2024],
            },
            v if v <= 0.3 => Self {
                temperature: 0.7,
                max_tokens: 2000,
                system_prompt_style: PromptStyle::Conversational,
                top_p: 0.9,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                seed_options: vec![1, 100, 1000],
            },
            v if v <= 0.6 => Self {
                temperature: 0.9,
                max_tokens: 3000,
                system_prompt_style: PromptStyle::Narrative,
                top_p: 0.95,
                frequency_penalty: -0.2,
                presence_penalty: -0.1,
                seed_options: vec![314, 271, 1618],
            },
            _ => Self {
                temperature: 1.1,
                max_tokens: 4000,
                system_prompt_style: PromptStyle::Creative,
                top_p: 0.95,
                frequency_penalty: -0.3,
                presence_penalty: -0.2,
                seed_options: vec![42, 314, 2718],
            },
        }
    }

    /// Blend two param sets with weight (0.0 = all self, 1.0 = all other)
    pub fn merge(&self, other: &ModelParams, weight: f64) -> ModelParams {
        let w = weight.clamp(0.0, 1.0);
        ModelParams {
            temperature: self.temperature * (1.0 - w) + other.temperature * w,
            max_tokens: ((self.max_tokens as f64) * (1.0 - w) + (other.max_tokens as f64) * w) as u32,
            system_prompt_style: if w < 0.5 { self.system_prompt_style } else { other.system_prompt_style },
            top_p: self.top_p * (1.0 - w) + other.top_p * w,
            frequency_penalty: self.frequency_penalty * (1.0 - w) + other.frequency_penalty * w,
            presence_penalty: self.presence_penalty * (1.0 - w) + other.presence_penalty * w,
            seed_options: if w < 0.5 { self.seed_options.clone() } else { other.seed_options.clone() },
        }
    }

    pub fn validate(&self) -> bool {
        self.temperature >= 0.0 && self.temperature <= 2.0
            && self.top_p >= 0.0 && self.top_p <= 1.0
            && self.frequency_penalty >= -2.0 && self.frequency_penalty <= 2.0
            && self.presence_penalty >= -2.0 && self.presence_penalty <= 2.0
    }
}

// ─── Room Gravity ─────────────────────────────────────────────────────────────

/// The gravity state for a specific room
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomGravity {
    pub room_id: String,
    pub gravity: Gravity,
    pub interaction_history: Vec<GravitySignal>,
    pub effective_params: ModelParams,
    pub decay_rate: f64,
}

impl RoomGravity {
    pub fn new(room_id: &str) -> Self {
        let gravity = Gravity::new();
        Self {
            room_id: room_id.to_string(),
            effective_params: ModelParams::from_gravity(&gravity),
            gravity,
            interaction_history: Vec::new(),
            decay_rate: 0.01,
        }
    }

    pub fn record_interaction(&mut self, signal: &GravitySignal) {
        let combined = signal.style_value() * signal.response_success;
        self.gravity.update(combined);
        self.effective_params = ModelParams::from_gravity(&self.gravity);
        self.interaction_history.push(signal.clone());
    }

    pub fn current_params(&self) -> ModelParams {
        self.effective_params.clone()
    }

    pub fn should_escalate(&self) -> bool {
        self.gravity.confidence < 0.3 && self.gravity.sample_count > 5
    }

    pub fn decay(&mut self) {
        self.gravity.value *= 1.0 - self.decay_rate;
        if self.gravity.value.abs() < 0.001 {
            self.gravity.value = 0.0;
        }
    }

    pub fn snapshot(&self) -> GravitySnapshot {
        GravitySnapshot {
            room_id: self.room_id.clone(),
            value: self.gravity.value,
            confidence: self.gravity.confidence,
            sample_count: self.gravity.sample_count,
            timestamp: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitySnapshot {
    pub room_id: String,
    pub value: f64,
    pub confidence: f64,
    pub sample_count: u32,
    pub timestamp: u64,
}

// ─── Gravity Cluster ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityCluster {
    pub center_gravity: f64,
    pub rooms: Vec<String>,
    pub label: String,
}

impl GravityCluster {
    pub fn new(center: f64) -> Self {
        Self {
            center_gravity: center,
            rooms: Vec::new(),
            label: String::new(),
        }
    }

    pub fn add_room(&mut self, room: &str) {
        if !self.rooms.contains(&room.to_string()) {
            self.rooms.push(room.to_string());
        }
    }
}

// ─── Gravity Field Summary ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityFieldSummary {
    pub room_count: usize,
    pub average_gravity: f64,
    pub field_entropy: f64,
    pub most_playful: Option<String>,
    pub most_serious: Option<String>,
    pub clusters: Vec<GravityCluster>,
}

// ─── Gravity Field ────────────────────────────────────────────────────────────

/// The field connecting all rooms
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GravityField {
    pub rooms: HashMap<String, RoomGravity>,
    pub tick: u64,
}

impl GravityField {
    pub fn new() -> Self {
        Self { rooms: HashMap::new(), tick: 0 }
    }

    pub fn register_room(&mut self, room_id: &str) {
        self.rooms.entry(room_id.to_string())
            .or_insert_with(|| RoomGravity::new(room_id));
    }

    pub fn record(&mut self, room_id: &str, signal: &GravitySignal) {
        self.register_room(room_id);
        if let Some(room) = self.rooms.get_mut(room_id) {
            room.record_interaction(signal);
        }
    }

    pub fn gravity_of(&self, room_id: &str) -> Option<f64> {
        self.rooms.get(room_id).map(|r| r.gravity.value)
    }

    pub fn closest_room(&self, target_gravity: f64) -> Option<String> {
        self.rooms.iter()
            .min_by(|a, b| {
                let da = (a.1.gravity.value - target_gravity).abs();
                let db = (b.1.gravity.value - target_gravity).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.clone())
    }

    /// Route a signal to rooms ordered by how well their gravity matches
    pub fn route_signal(&self, signal: &GravitySignal) -> Vec<String> {
        let target = signal.style_value();
        let mut ranked: Vec<(String, f64)> = self.rooms.iter()
            .map(|(id, r)| (id.clone(), (r.gravity.value - target).abs()))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().map(|(id, _)| id).collect()
    }

    pub fn gravity_map(&self) -> HashMap<String, f64> {
        self.rooms.iter().map(|(id, r)| (id.clone(), r.gravity.value)).collect()
    }

    pub fn gravity_gradient(&self, room_id: &str) -> Vec<(String, f64)> {
        let base = match self.rooms.get(room_id) {
            Some(r) => r.gravity.value,
            None => return Vec::new(),
        };
        let mut v: Vec<(String, f64)> = self.rooms.iter()
            .filter(|(id, _)| id.as_str() != room_id)
            .map(|(id, r)| (id.clone(), (r.gravity.value - base).abs()))
            .collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Shannon-style entropy across room gravities
    pub fn field_entropy(&self) -> f64 {
        if self.rooms.is_empty() { return 0.0; }
        let n = self.rooms.len() as f64;
        // Bin into 10 buckets from -1 to 1
        let bucket_count = 10usize;
        let mut buckets = vec![0usize; bucket_count];
        for r in self.rooms.values() {
            let idx = (((r.gravity.value + 1.0) / 2.0) * (bucket_count as f64 - 1.0)).round() as usize;
            let idx = idx.min(bucket_count - 1);
            buckets[idx] += 1;
        }
        let mut entropy = 0.0;
        for &count in &buckets {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    pub fn tick(&mut self) {
        self.tick += 1;
        for room in self.rooms.values_mut() {
            room.decay();
        }
        // Prune rooms that have decayed to zero and have no history
        self.rooms.retain(|_, r| !(r.gravity.value == 0.0 && r.interaction_history.is_empty() && r.gravity.sample_count > 10));
    }

    pub fn field_summary(&self) -> GravityFieldSummary {
        let avg = if self.rooms.is_empty() { 0.0 } else {
            self.rooms.values().map(|r| r.gravity.value).sum::<f64>() / self.rooms.len() as f64
        };
        let most_playful = self.rooms.iter()
            .filter(|(_, r)| r.gravity.is_playful())
            .max_by(|a, b| a.1.gravity.value.partial_cmp(&b.1.gravity.value).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.clone());
        let most_serious = self.rooms.iter()
            .filter(|(_, r)| r.gravity.is_serious())
            .min_by(|a, b| a.1.gravity.value.partial_cmp(&b.1.gravity.value).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id.clone());
        GravityFieldSummary {
            room_count: self.rooms.len(),
            average_gravity: avg,
            field_entropy: self.field_entropy(),
            most_playful,
            most_serious,
            clusters: self.compute_clusters(),
        }
    }

    fn compute_clusters(&self) -> Vec<GravityCluster> {
        let mut clusters = Vec::new();
        let centers = [-0.8, -0.4, 0.0, 0.4, 0.8];
        for &center in &centers {
            let mut cluster = GravityCluster::new(center);
            cluster.label = match center {
                c if c < -0.6 => "Very Serious".into(),
                c if c < -0.2 => "Somewhat Serious".into(),
                c if c <= 0.2 => "Balanced".into(),
                c if c <= 0.6 => "Somewhat Playful".into(),
                _ => "Very Playful".into(),
            };
            for (id, r) in &self.rooms {
                if (r.gravity.value - center).abs() <= 0.2 {
                    cluster.add_room(id);
                }
            }
            if !cluster.rooms.is_empty() {
                clusters.push(cluster);
            }
        }
        clusters
    }
}

// ─── Phone A Friend ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneAFriend {
    pub large_model: String,
    pub small_model: String,
    pub escalation_threshold: f64,
    pub call_count: u32,
    pub call_limit: u32,
}

impl PhoneAFriend {
    pub fn new(large: &str, small: &str) -> Self {
        Self {
            large_model: large.to_string(),
            small_model: small.to_string(),
            escalation_threshold: 0.3,
            call_count: 0,
            call_limit: 10,
        }
    }

    pub fn should_call(&self, room_gravity: &RoomGravity) -> bool {
        room_gravity.gravity.confidence < self.escalation_threshold
            && self.call_count < self.call_limit
            && room_gravity.gravity.sample_count > 3
    }

    pub fn record_call(&mut self) {
        self.call_count += 1;
    }

    pub fn calls_remaining(&self) -> u32 {
        self.call_limit.saturating_sub(self.call_count)
    }

    pub fn simulate_responses(&self, prompt: &str, variations: usize) -> Vec<SimulatedResponse> {
        (0..variations).map(|i| {
            SimulatedResponse::new(prompt, &format!("variation_{} of {}", i + 1, prompt))
        }).collect()
    }
}

// ─── Simulated Response ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedResponse {
    pub prompt: String,
    pub response: String,
    pub gravity_match: f64,
    pub user_style_match: f64,
    pub quality_score: f64,
}

impl SimulatedResponse {
    pub fn new(prompt: &str, response: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            response: response.to_string(),
            gravity_match: 0.5,
            user_style_match: 0.5,
            quality_score: 0.5,
        }
    }
}

// ─── Mandelbrot Zoom ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    pub name: String,
    pub depth: u32,
    pub complexity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MandelbrotZoom {
    pub room_id: String,
    pub current_depth: u32,
    pub min_tile_size: f64,
    pub decompositions: Vec<Decomposition>,
}

impl MandelbrotZoom {
    pub fn new(room_id: &str) -> Self {
        Self {
            room_id: room_id.to_string(),
            current_depth: 1,
            min_tile_size: 1.0,
            decompositions: Vec::new(),
        }
    }

    /// Estimate irreducible complexity from interactions and patterns
    pub fn measure_complexity(&mut self, interactions: usize, unique_patterns: usize) -> f64 {
        let interaction_density = if interactions == 0 { 0.0 } else {
            unique_patterns as f64 / interactions as f64
        };
        self.min_tile_size = (1.0 / (self.current_depth as f64)).max(0.001);
        interaction_density * self.min_tile_size
    }

    pub fn should_zoom_in(&self, error_rate: f64) -> bool {
        error_rate > 0.3 && self.current_depth < 20
    }

    pub fn should_zoom_out(&self, efficiency: f64) -> bool {
        efficiency > 0.9 && self.current_depth > 1
    }

    pub fn zoom_in(&mut self) -> Vec<Decomposition> {
        self.current_depth += 1;
        self.min_tile_size = (1.0 / (self.current_depth as f64)).max(0.001);
        let subs = ["α", "β", "γ", "δ"];
        let decomp: Vec<Decomposition> = subs.iter().map(|&s| Decomposition {
            name: format!("{}_d{}_{}", self.room_id, self.current_depth, s),
            depth: self.current_depth,
            complexity: self.min_tile_size,
        }).collect();
        self.decompositions.extend(decomp.clone());
        decomp
    }

    pub fn zoom_out(&mut self) {
        if self.current_depth > 1 {
            self.current_depth -= 1;
            self.min_tile_size = (1.0 / (self.current_depth as f64)).max(0.001);
            self.decompositions.retain(|d| d.depth <= self.current_depth);
        }
    }

    pub fn current_tile_size(&self) -> f64 {
        self.min_tile_size
    }

    pub fn depth_report(&self) -> String {
        format!(
            "Room {} — depth: {}, tile_size: {:.4}, decompositions: {}",
            self.room_id, self.current_depth, self.min_tile_size, self.decompositions.len()
        )
    }
}

// ─── Progressive Generation ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveGeneration {
    pub room_id: String,
    pub generation_level: u32,
    pub success_history: Vec<bool>,
    pub model_usage: HashMap<String, u32>,
}

impl ProgressiveGeneration {
    pub fn new(room_id: &str) -> Self {
        Self {
            room_id: room_id.to_string(),
            generation_level: 1,
            success_history: Vec::new(),
            model_usage: HashMap::new(),
        }
    }

    /// Pick model based on generation level and gravity
    pub fn pick_model(&self, gravity: &Gravity, phone: &PhoneAFriend) -> String {
        if self.generation_level <= 2 || gravity.confidence < 0.5 {
            phone.large_model.clone()
        } else {
            phone.small_model.clone()
        }
    }

    pub fn record_success(&mut self, model: &str, success: bool) {
        *self.model_usage.entry(model.to_string()).or_insert(0) += 1;
        self.success_history.push(success);
    }

    pub fn promote(&mut self) {
        if self.generation_level < 5 {
            self.generation_level += 1;
        }
    }

    pub fn demote(&mut self) {
        if self.generation_level > 1 {
            self.generation_level -= 1;
        }
    }

    pub fn model_efficiency(&self) -> f64 {
        if self.success_history.is_empty() { return 0.0; }
        let successes = self.success_history.iter().filter(|&&s| s).count();
        successes as f64 / self.success_history.len() as f64
    }

    pub fn generation_report(&self) -> String {
        format!(
            "Room {} — level: {}/5, efficiency: {:.1}%, history: {} interactions",
            self.room_id,
            self.generation_level,
            self.model_efficiency() * 100.0,
            self.success_history.len()
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Gravity tests ──

    #[test]
    fn gravity_new_is_neutral() {
        let g = Gravity::new();
        assert_eq!(g.value, 0.0);
        assert_eq!(g.confidence, 0.0);
        assert_eq!(g.sample_count, 0);
    }

    #[test]
    fn gravity_from_value_clamps() {
        let g = Gravity::from_value(2.0);
        assert_eq!(g.value, 1.0);
        let g = Gravity::from_value(-3.0);
        assert_eq!(g.value, -1.0);
        let g = Gravity::from_value(0.5);
        assert_eq!(g.value, 0.5);
    }

    #[test]
    fn gravity_update_moves_value() {
        let mut g = Gravity::new();
        g.update(0.8);
        assert!(g.value > 0.0);
        assert_eq!(g.sample_count, 1);
    }

    #[test]
    fn gravity_update_clamps() {
        let mut g = Gravity::from_value(0.9);
        g.update(1.5);
        assert!(g.value <= 1.0);
    }

    #[test]
    fn gravity_is_playful() {
        assert!(Gravity::from_value(0.5).is_playful());
        assert!(!Gravity::from_value(0.2).is_playful());
        assert!(!Gravity::from_value(-0.5).is_playful());
    }

    #[test]
    fn gravity_is_serious() {
        assert!(Gravity::from_value(-0.5).is_serious());
        assert!(!Gravity::from_value(-0.2).is_serious());
        assert!(!Gravity::from_value(0.5).is_serious());
    }

    #[test]
    fn gravity_is_balanced() {
        assert!(Gravity::from_value(0.0).is_balanced());
        assert!(Gravity::from_value(0.3).is_balanced());
        assert!(Gravity::from_value(-0.3).is_balanced());
        assert!(!Gravity::from_value(0.4).is_balanced());
    }

    #[test]
    fn gravity_distance_to() {
        let a = Gravity::from_value(0.5);
        let b = Gravity::from_value(-0.3);
        assert!((a.distance_to(&b) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn gravity_nudge_toward() {
        let mut g = Gravity::from_value(-0.5);
        let target = Gravity::from_value(0.5);
        g.nudge_toward(&target, 0.5);
        assert!(g.value > -0.5);
        assert!(g.value < 0.5);
    }

    #[test]
    fn gravity_nudge_full() {
        let mut g = Gravity::from_value(0.0);
        let target = Gravity::from_value(1.0);
        g.nudge_toward(&target, 1.0);
        assert!((g.value - 1.0).abs() < 1e-10);
    }

    // ── GravitySignal tests ──

    #[test]
    fn signal_style_values() {
        assert_eq!(GravitySignal { user_style: UserStyle::Playful, response_success: 1.0, timestamp: 0, context: String::new() }.style_value(), 0.8);
        assert_eq!(GravitySignal { user_style: UserStyle::Precise, response_success: 1.0, timestamp: 0, context: String::new() }.style_value(), -0.8);
        assert_eq!(GravitySignal { user_style: UserStyle::Narrative, response_success: 1.0, timestamp: 0, context: String::new() }.style_value(), 0.3);
        assert_eq!(GravitySignal { user_style: UserStyle::Direct, response_success: 1.0, timestamp: 0, context: String::new() }.style_value(), -0.3);
        assert_eq!(GravitySignal { user_style: UserStyle::Socratic, response_success: 1.0, timestamp: 0, context: String::new() }.style_value(), 0.0);
        assert_eq!(GravitySignal { user_style: UserStyle::Mixed, response_success: 1.0, timestamp: 0, context: String::new() }.style_value(), 0.0);
    }

    #[test]
    fn signal_detects_playful() {
        let s = GravitySignal::from_text("lol that's hilarious 😂!!");
        assert_eq!(s.user_style, UserStyle::Playful);
    }

    #[test]
    fn signal_detects_precise() {
        let s = GravitySignal::from_text("fix the bug in the calculate function precisely");
        assert_eq!(s.user_style, UserStyle::Precise);
    }

    #[test]
    fn signal_detects_socratic() {
        let s = GravitySignal::from_text("why? how? what if we did it differently? could you explain?");
        assert_eq!(s.user_style, UserStyle::Socratic);
    }

    #[test]
    fn signal_detects_direct() {
        let s = GravitySignal::from_text("do it");
        assert_eq!(s.user_style, UserStyle::Direct);
    }

    #[test]
    fn signal_detects_mixed_for_neutral() {
        let s = GravitySignal::from_text("the weather is nice today and I went for a walk");
        assert_eq!(s.user_style, UserStyle::Mixed);
    }

    // ── ModelParams tests ──

    #[test]
    fn params_technical_for_very_serious() {
        let g = Gravity::from_value(-0.8);
        let p = ModelParams::from_gravity(&g);
        assert_eq!(p.system_prompt_style, PromptStyle::Technical);
        assert!((p.temperature - 0.3).abs() < 1e-10);
    }

    #[test]
    fn params_minimal_for_somewhat_serious() {
        let g = Gravity::from_value(-0.4);
        let p = ModelParams::from_gravity(&g);
        assert_eq!(p.system_prompt_style, PromptStyle::Minimal);
    }

    #[test]
    fn params_conversational_for_balanced() {
        let g = Gravity::from_value(0.0);
        let p = ModelParams::from_gravity(&g);
        assert_eq!(p.system_prompt_style, PromptStyle::Conversational);
    }

    #[test]
    fn params_narrative_for_somewhat_playful() {
        let g = Gravity::from_value(0.5);
        let p = ModelParams::from_gravity(&g);
        assert_eq!(p.system_prompt_style, PromptStyle::Narrative);
    }

    #[test]
    fn params_creative_for_very_playful() {
        let g = Gravity::from_value(0.8);
        let p = ModelParams::from_gravity(&g);
        assert_eq!(p.system_prompt_style, PromptStyle::Creative);
        assert!((p.temperature - 1.1).abs() < 1e-10);
    }

    #[test]
    fn params_merge_weight_zero() {
        let a = ModelParams::from_gravity(&Gravity::from_value(-0.8));
        let b = ModelParams::from_gravity(&Gravity::from_value(0.8));
        let m = a.merge(&b, 0.0);
        assert_eq!(m.system_prompt_style, a.system_prompt_style);
        assert!((m.temperature - a.temperature).abs() < 1e-10);
    }

    #[test]
    fn params_merge_weight_one() {
        let a = ModelParams::from_gravity(&Gravity::from_value(-0.8));
        let b = ModelParams::from_gravity(&Gravity::from_value(0.8));
        let m = a.merge(&b, 1.0);
        assert_eq!(m.system_prompt_style, b.system_prompt_style);
    }

    #[test]
    fn params_validate_good() {
        let p = ModelParams::from_gravity(&Gravity::from_value(0.0));
        assert!(p.validate());
    }

    #[test]
    fn prompt_style_text() {
        assert!(!PromptStyle::Technical.prompt_text().is_empty());
        assert!(!PromptStyle::Creative.prompt_text().is_empty());
    }

    // ── RoomGravity tests ──

    #[test]
    fn room_gravity_new() {
        let r = RoomGravity::new("room1");
        assert_eq!(r.room_id, "room1");
        assert!(r.gravity.is_balanced());
        assert!(r.interaction_history.is_empty());
    }

    #[test]
    fn room_gravity_record_interaction() {
        let mut r = RoomGravity::new("room1");
        let signal = GravitySignal { user_style: UserStyle::Playful, response_success: 1.0, timestamp: 1, context: "lol".into() };
        r.record_interaction(&signal);
        assert_eq!(r.interaction_history.len(), 1);
        assert!(r.gravity.value > 0.0);
    }

    #[test]
    fn room_gravity_should_escalate_low_confidence() {
        let mut r = RoomGravity::new("room1");
        r.gravity.confidence = 0.1;
        r.gravity.sample_count = 10;
        assert!(r.should_escalate());
    }

    #[test]
    fn room_gravity_should_not_escalate_few_samples() {
        let mut r = RoomGravity::new("room1");
        r.gravity.confidence = 0.1;
        r.gravity.sample_count = 2;
        assert!(!r.should_escalate());
    }

    #[test]
    fn room_gravity_decay() {
        let mut r = RoomGravity::new("room1");
        r.gravity.value = 0.5;
        r.decay();
        assert!(r.gravity.value < 0.5);
        assert!(r.gravity.value > 0.0);
    }

    #[test]
    fn room_gravity_snapshot() {
        let r = RoomGravity::new("room1");
        let s = r.snapshot();
        assert_eq!(s.room_id, "room1");
    }

    // ── GravityField tests ──

    #[test]
    fn field_register_room() {
        let mut f = GravityField::new();
        f.register_room("r1");
        assert!(f.rooms.contains_key("r1"));
        f.register_room("r1"); // no-op
        assert_eq!(f.rooms.len(), 1);
    }

    #[test]
    fn field_record() {
        let mut f = GravityField::new();
        let sig = GravitySignal { user_style: UserStyle::Precise, response_success: 1.0, timestamp: 0, context: String::new() };
        f.record("r1", &sig);
        assert!(f.gravity_of("r1").unwrap() < 0.0);
    }

    #[test]
    fn field_gravity_of_missing() {
        let f = GravityField::new();
        assert!(f.gravity_of("missing").is_none());
    }

    #[test]
    fn field_closest_room() {
        let mut f = GravityField::new();
        f.register_room("serious");
        f.rooms.get_mut("serious").unwrap().gravity.value = -0.8;
        f.register_room("playful");
        f.rooms.get_mut("playful").unwrap().gravity.value = 0.8;
        assert_eq!(f.closest_room(-0.9).unwrap(), "serious");
        assert_eq!(f.closest_room(0.9).unwrap(), "playful");
    }

    #[test]
    fn field_route_signal() {
        let mut f = GravityField::new();
        f.register_room("serious");
        f.rooms.get_mut("serious").unwrap().gravity.value = -0.8;
        f.register_room("playful");
        f.rooms.get_mut("playful").unwrap().gravity.value = 0.8;
        let sig = GravitySignal { user_style: UserStyle::Playful, response_success: 1.0, timestamp: 0, context: String::new() };
        let route = f.route_signal(&sig);
        assert_eq!(route[0], "playful");
    }

    #[test]
    fn field_gravity_map() {
        let mut f = GravityField::new();
        f.register_room("r1");
        f.rooms.get_mut("r1").unwrap().gravity.value = 0.5;
        let map = f.gravity_map();
        assert_eq!(map["r1"], 0.5);
    }

    #[test]
    fn field_gravity_gradient() {
        let mut f = GravityField::new();
        f.register_room("r0");
        f.rooms.get_mut("r0").unwrap().gravity.value = 0.0;
        f.register_room("r1");
        f.rooms.get_mut("r1").unwrap().gravity.value = 0.1;
        f.register_room("r2");
        f.rooms.get_mut("r2").unwrap().gravity.value = 0.9;
        let grad = f.gravity_gradient("r0");
        assert_eq!(grad.len(), 2);
        assert!((grad[0].1 - 0.1).abs() < 1e-10); // r1 closest
        assert!((grad[1].1 - 0.9).abs() < 1e-10); // r2 farthest
    }

    #[test]
    fn field_entropy_diverse() {
        let mut f = GravityField::new();
        for i in 0..10 {
            let id = format!("r{}", i);
            f.register_room(&id);
            f.rooms.get_mut(&id).unwrap().gravity.value = -1.0 + (i as f64) * 0.22;
        }
        let e = f.field_entropy();
        assert!(e > 2.0); // high entropy
    }

    #[test]
    fn field_entropy_uniform() {
        let mut f = GravityField::new();
        for i in 0..5 {
            let id = format!("r{}", i);
            f.register_room(&id);
            f.rooms.get_mut(&id).unwrap().gravity.value = 0.0;
        }
        let e = f.field_entropy();
        assert!(e.abs() < 0.01); // all same bucket, very low entropy
    }

    #[test]
    fn field_tick_decay() {
        let mut f = GravityField::new();
        f.register_room("r1");
        f.rooms.get_mut("r1").unwrap().gravity.value = 0.5;
        f.tick();
        assert!(f.rooms.get("r1").unwrap().gravity.value < 0.5);
    }

    #[test]
    fn field_summary() {
        let mut f = GravityField::new();
        f.register_room("serious");
        f.rooms.get_mut("serious").unwrap().gravity.value = -0.8;
        f.register_room("playful");
        f.rooms.get_mut("playful").unwrap().gravity.value = 0.8;
        let s = f.field_summary();
        assert_eq!(s.room_count, 2);
        assert_eq!(s.most_serious.unwrap(), "serious");
        assert_eq!(s.most_playful.unwrap(), "playful");
    }

    #[test]
    fn field_summary_empty() {
        let f = GravityField::new();
        let s = f.field_summary();
        assert_eq!(s.room_count, 0);
        assert!(s.most_playful.is_none());
    }

    // ── GravityCluster tests ──

    #[test]
    fn cluster_new() {
        let c = GravityCluster::new(0.5);
        assert_eq!(c.center_gravity, 0.5);
        assert!(c.rooms.is_empty());
    }

    #[test]
    fn cluster_add_room() {
        let mut c = GravityCluster::new(0.0);
        c.add_room("r1");
        c.add_room("r1"); // duplicate
        assert_eq!(c.rooms.len(), 1);
    }

    // ── PhoneAFriend tests ──

    #[test]
    fn phone_a_friend_new() {
        let p = PhoneAFriend::new("gpt-4", "gpt-3.5");
        assert_eq!(p.large_model, "gpt-4");
        assert_eq!(p.small_model, "gpt-3.5");
        assert_eq!(p.calls_remaining(), 10);
    }

    #[test]
    fn phone_should_call_low_confidence() {
        let p = PhoneAFriend::new("large", "small");
        let mut r = RoomGravity::new("r1");
        r.gravity.confidence = 0.1;
        r.gravity.sample_count = 10;
        assert!(p.should_call(&r));
    }

    #[test]
    fn phone_should_not_call_high_confidence() {
        let p = PhoneAFriend::new("large", "small");
        let r = RoomGravity::new("r1");
        assert!(!p.should_call(&r)); // confidence 0 but sample_count 0
    }

    #[test]
    fn phone_record_call() {
        let mut p = PhoneAFriend::new("large", "small");
        p.record_call();
        assert_eq!(p.call_count, 1);
        assert_eq!(p.calls_remaining(), 9);
    }

    #[test]
    fn phone_simulate_responses() {
        let p = PhoneAFriend::new("large", "small");
        let resps = p.simulate_responses("hello", 3);
        assert_eq!(resps.len(), 3);
        assert_eq!(resps[0].prompt, "hello");
    }

    #[test]
    fn phone_call_limit() {
        let mut p = PhoneAFriend::new("large", "small");
        p.call_limit = 2;
        p.record_call();
        p.record_call();
        let mut r = RoomGravity::new("r1");
        r.gravity.confidence = 0.1;
        r.gravity.sample_count = 10;
        assert!(!p.should_call(&r)); // limit reached
    }

    // ── SimulatedResponse tests ──

    #[test]
    fn simulated_response_new() {
        let sr = SimulatedResponse::new("prompt", "response");
        assert_eq!(sr.prompt, "prompt");
        assert_eq!(sr.response, "response");
        assert!((sr.gravity_match - 0.5).abs() < 1e-10);
    }

    // ── MandelbrotZoom tests ──

    #[test]
    fn mandelbrot_new() {
        let m = MandelbrotZoom::new("r1");
        assert_eq!(m.current_depth, 1);
        assert!((m.min_tile_size - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mandelbrot_measure_complexity() {
        let mut m = MandelbrotZoom::new("r1");
        let c = m.measure_complexity(100, 50);
        assert!(c > 0.0);
    }

    #[test]
    fn mandelbrot_should_zoom_in() {
        let m = MandelbrotZoom::new("r1");
        assert!(m.should_zoom_in(0.5));
        assert!(!m.should_zoom_in(0.1));
    }

    #[test]
    fn mandelbrot_should_zoom_out() {
        let mut m = MandelbrotZoom::new("r1");
        m.zoom_in(); // depth 2 so we can zoom out
        assert!(m.should_zoom_out(0.95));
        assert!(!m.should_zoom_out(0.5));
    }

    #[test]
    fn mandelbrot_zoom_in_produces_decompositions() {
        let mut m = MandelbrotZoom::new("r1");
        let d = m.zoom_in();
        assert_eq!(d.len(), 4);
        assert_eq!(m.current_depth, 2);
        assert!(m.min_tile_size < 1.0);
    }

    #[test]
    fn mandelbrot_zoom_out_reduces_depth() {
        let mut m = MandelbrotZoom::new("r1");
        m.zoom_in();
        m.zoom_in();
        assert_eq!(m.current_depth, 3);
        m.zoom_out();
        assert_eq!(m.current_depth, 2);
    }

    #[test]
    fn mandelbrot_zoom_out_min_depth() {
        let mut m = MandelbrotZoom::new("r1");
        m.zoom_out(); // already at 1
        assert_eq!(m.current_depth, 1);
    }

    #[test]
    fn mandelbrot_depth_report() {
        let m = MandelbrotZoom::new("r1");
        let report = m.depth_report();
        assert!(report.contains("r1"));
        assert!(report.contains("depth: 1"));
    }

    #[test]
    fn mandelbrot_max_depth() {
        let mut m = MandelbrotZoom::new("r1");
        m.current_depth = 20;
        assert!(!m.should_zoom_in(0.5)); // at max depth
    }

    // ── ProgressiveGeneration tests ──

    #[test]
    fn progressive_new() {
        let pg = ProgressiveGeneration::new("r1");
        assert_eq!(pg.generation_level, 1);
    }

    #[test]
    fn progressive_pick_model_early() {
        let pg = ProgressiveGeneration::new("r1");
        let phone = PhoneAFriend::new("large", "small");
        let g = Gravity::from_value(0.5);
        assert_eq!(pg.pick_model(&g, &phone), "large"); // level 1
    }

    #[test]
    fn progressive_pick_model_late() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.generation_level = 4;
        let phone = PhoneAFriend::new("large", "small");
        let g = Gravity { value: 0.5, confidence: 0.9, sample_count: 50 };
        assert_eq!(pg.pick_model(&g, &phone), "small");
    }

    #[test]
    fn progressive_record_success() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.record_success("large", true);
        pg.record_success("small", false);
        assert_eq!(pg.success_history.len(), 2);
        assert_eq!(pg.model_usage["large"], 1);
        assert_eq!(pg.model_usage["small"], 1);
    }

    #[test]
    fn progressive_promote() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.promote();
        assert_eq!(pg.generation_level, 2);
    }

    #[test]
    fn progressive_promote_max() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.generation_level = 5;
        pg.promote();
        assert_eq!(pg.generation_level, 5); // stays at 5
    }

    #[test]
    fn progressive_demote() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.generation_level = 3;
        pg.demote();
        assert_eq!(pg.generation_level, 2);
    }

    #[test]
    fn progressive_demote_min() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.demote();
        assert_eq!(pg.generation_level, 1);
    }

    #[test]
    fn progressive_efficiency() {
        let mut pg = ProgressiveGeneration::new("r1");
        pg.record_success("m", true);
        pg.record_success("m", true);
        pg.record_success("m", false);
        assert!((pg.model_efficiency() - (2.0/3.0)).abs() < 1e-10);
    }

    #[test]
    fn progressive_efficiency_empty() {
        let pg = ProgressiveGeneration::new("r1");
        assert_eq!(pg.model_efficiency(), 0.0);
    }

    #[test]
    fn progressive_generation_report() {
        let pg = ProgressiveGeneration::new("r1");
        let report = pg.generation_report();
        assert!(report.contains("r1"));
    }

    // ── Serde round-trip tests ──

    #[test]
    fn serde_gravity() {
        let g = Gravity::from_value(0.7);
        let json = serde_json::to_string(&g).unwrap();
        let g2: Gravity = serde_json::from_str(&json).unwrap();
        assert_eq!(g, g2);
    }

    #[test]
    fn serde_model_params() {
        let p = ModelParams::from_gravity(&Gravity::from_value(-0.5));
        let json = serde_json::to_string(&p).unwrap();
        let p2: ModelParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn serde_gravity_field() {
        let mut f = GravityField::new();
        f.register_room("r1");
        let json = serde_json::to_string(&f).unwrap();
        let f2: GravityField = serde_json::from_str(&json).unwrap();
        assert_eq!(f2.rooms.len(), 1);
    }

    #[test]
    fn serde_user_style() {
        let s = UserStyle::Playful;
        let json = serde_json::to_string(&s).unwrap();
        let s2: UserStyle = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn serde_phone_a_friend() {
        let p = PhoneAFriend::new("big", "small");
        let json = serde_json::to_string(&p).unwrap();
        let p2: PhoneAFriend = serde_json::from_str(&json).unwrap();
        assert_eq!(p2.large_model, "big");
    }

    #[test]
    fn serde_mandelbrot() {
        let mut m = MandelbrotZoom::new("r1");
        m.zoom_in();
        let json = serde_json::to_string(&m).unwrap();
        let m2: MandelbrotZoom = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.current_depth, 2);
    }

    // ── Integration test ──

    #[test]
    fn full_workflow() {
        let mut field = GravityField::new();
        field.register_room("dev-room");
        field.register_room("chat-room");
        field.register_room("story-room");

        // Dev room gets precise signals
        let precise = GravitySignal { user_style: UserStyle::Precise, response_success: 0.9, timestamp: 1, context: "debug this".into() };
        for _ in 0..5 {
            field.record("dev-room", &precise);
        }

        // Chat room gets playful signals
        let playful = GravitySignal { user_style: UserStyle::Playful, response_success: 0.8, timestamp: 2, context: "lol haha".into() };
        for _ in 0..5 {
            field.record("chat-room", &playful);
        }

        // Story room gets narrative signals
        let narrative = GravitySignal { user_style: UserStyle::Narrative, response_success: 0.7, timestamp: 3, context: "once upon a time".into() };
        for _ in 0..5 {
            field.record("story-room", &narrative);
        }

        let dev_g = field.gravity_of("dev-room").unwrap();
        let chat_g = field.gravity_of("chat-room").unwrap();
        let story_g = field.gravity_of("story-room").unwrap();

        assert!(dev_g < -0.2, "dev-room should be serious, got {}", dev_g);
        assert!(chat_g > 0.2, "chat-room should be playful, got {}", chat_g);
        assert!(story_g > 0.0, "story-room should be slightly positive, got {}", story_g);

        // Routing: a playful signal should route to chat-room first
        let route = field.route_signal(&playful);
        assert_eq!(route[0], "chat-room");

        // Summary
        let summary = field.field_summary();
        assert_eq!(summary.room_count, 3);
        assert_eq!(summary.most_serious.unwrap(), "dev-room");
        assert_eq!(summary.most_playful.unwrap(), "chat-room");

        // Phone-a-friend
        let _phone = PhoneAFriend::new("gpt-4", "gpt-3.5");
        let dev_room = field.rooms.get("dev-room").unwrap();
        let params = dev_room.current_params();
        assert!(params.validate());

        // Mandelbrot
        let mut zoom = MandelbrotZoom::new("dev-room");
        let complexity = zoom.measure_complexity(100, 80);
        assert!(complexity > 0.0);
        if zoom.should_zoom_in(0.5) {
            let decs = zoom.zoom_in();
            assert_eq!(decs.len(), 4);
        }

        // Progressive generation
        let mut pg = ProgressiveGeneration::new("dev-room");
        pg.record_success("gpt-4", true);
        pg.record_success("gpt-4", true);
        pg.record_success("gpt-3.5", false);
        assert!(pg.model_efficiency() > 0.5);
    }
}
