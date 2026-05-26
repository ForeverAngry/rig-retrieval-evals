//! Captured-run shape consumed by graders.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// One agent run, captured for grading.
///
/// `Transcript` is intentionally agent-shape-agnostic: it mirrors the
/// structured fields that the OpenAI Codex `--json` trace and Anthropic
/// transcript model both expose, plus a `skill_invoked` hint that the
/// runner populates when the agent's router selects a skill explicitly.
///
/// The runner is responsible for filling the fields it can. Graders treat
/// missing optionals as "unknown" rather than asserting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transcript {
    /// The prompt that was sent to the agent.
    pub prompt: String,
    /// The agent's final response text. May be empty.
    #[serde(default)]
    pub final_output: String,
    /// Tool calls observed during the run, in order.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Token / cost telemetry, if the runner can attribute it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Wall-clock duration of the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed: Option<Duration>,
    /// Number of conversation turns (assistant messages) the agent took.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<usize>,
    /// Name of the skill the router selected, if the runner exposes it.
    ///
    /// Populate this when you can observe the dispatch decision (e.g. via
    /// `rig-compose`'s `CoordinatorAgent` or by listening to a
    /// `skill.selected` event from `rig-tap`). Leave `None` if your
    /// agent does not surface a routing signal — trigger graders are then
    /// skipped without failing the task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_invoked: Option<String>,
    /// Optional free-form metadata for downstream tooling.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// One tool invocation captured in a [`Transcript`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool identifier (e.g. `"search_web"`, `"npm install"`).
    pub name: String,
    /// JSON-encoded arguments passed to the tool. `Value::Null` if the
    /// runner cannot capture arguments.
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Whether the tool reported success. `None` when the runner does not
    /// distinguish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
}

impl ToolCall {
    /// Construct a [`ToolCall`] with a JSON-null argument blob and unknown
    /// success.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: serde_json::Value::Null,
            ok: None,
        }
    }
}

/// Aggregated token / cost telemetry for a single trial.
///
/// Matches the shape emitted by `rig-model-catalog`'s `MetaHook` so wiring the
/// two together is just a `From` impl on the caller side.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Total input tokens billed across the trial.
    #[serde(default)]
    pub input_tokens: u64,
    /// Total output tokens billed across the trial.
    #[serde(default)]
    pub output_tokens: u64,
    /// Optional per-trial cost in USD. Populate from
    /// `rig_model_catalog::PricingTable::cost_for_usage` if you need a
    /// monetary efficiency dimension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
}

impl Usage {
    /// Sum of input and output tokens.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}
