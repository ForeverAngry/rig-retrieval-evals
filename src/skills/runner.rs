//! User-supplied agent invocation contract.

use std::future::Future;

use crate::error::Result;
use crate::skills::task::SkillTask;
use crate::skills::transcript::Transcript;

/// Runs one trial of one task and returns the captured [`Transcript`].
///
/// Implementations own whatever agent / harness is under test. The
/// [`SkillHarness`](crate::skills::SkillHarness) calls
/// [`AgentRunner::run`] exactly once per `(task, trial)` pair, so the
/// implementation is the right place to enforce isolation: a fresh memory,
/// a clean working directory, a new agent instance — anything the
/// Anthropic and Claude playbooks call "clean context per trial."
///
/// ```no_run
/// use std::future::Future;
/// use rig_retrieval_evals::skills::{AgentRunner, SkillTask, Transcript};
/// use rig_retrieval_evals::Result;
///
/// struct EchoRunner;
///
/// impl AgentRunner for EchoRunner {
///     fn run(
///         &self,
///         task: &SkillTask,
///         _trial: usize,
///     ) -> impl Future<Output = Result<Transcript>> + Send {
///         let prompt = task.prompt.clone();
///         async move {
///             Ok(Transcript {
///                 prompt: prompt.clone(),
///                 final_output: prompt,
///                 ..Default::default()
///             })
///         }
///     }
/// }
/// ```
pub trait AgentRunner: Send + Sync {
    /// Execute one trial of `task` (the `trial` index is 0-based) and
    /// return the captured transcript.
    fn run(
        &self,
        task: &SkillTask,
        trial: usize,
    ) -> impl Future<Output = Result<Transcript>> + Send;
}
