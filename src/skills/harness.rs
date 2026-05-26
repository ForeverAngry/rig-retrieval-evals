//! Driver and report types for skill evaluation.

use std::collections::{BTreeMap, BTreeSet};

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::error::{Error, Result};
use crate::report::{MetricReport, ReliabilityReport};
use crate::skills::grader::{AsyncGrader, GraderOutcome};
use crate::skills::runner::AgentRunner;
use crate::skills::task::SkillTaskSet;
use crate::skills::transcript::Transcript;

/// Drives `(tasks × trials)` runs and aggregates grader outcomes.
///
/// The harness owns a fixed grader registry and a runner. Each `run` call
/// executes every task once per trial, applies every grader the task
/// references, and produces a [`SkillEvalReport`] with one
/// [`MetricReport`] per `(grader_id, trial)` and one
/// [`ReliabilityReport`] per grader.
///
/// ```no_run
/// # use std::future::Future;
/// # use rig_retrieval_evals::Result;
/// # use rig_retrieval_evals::skills::{
/// #     AgentRunner, AsyncGrader, ContainsGrader, SkillHarness, SkillTask, SkillTaskSet,
/// #     Transcript,
/// # };
/// # struct R;
/// # impl AgentRunner for R {
/// #     fn run(
/// #         &self,
/// #         task: &SkillTask,
/// #         _: usize,
/// #     ) -> impl Future<Output = Result<Transcript>> + Send {
/// #         let p = task.prompt.clone();
/// #         async move { Ok(Transcript { prompt: p.clone(), final_output: p, ..Default::default() }) }
/// #     }
/// # }
/// # async fn run() -> Result<()> {
/// let graders: Vec<Box<dyn AsyncGrader>> =
///     vec![Box::new(ContainsGrader::present("greets", "hello"))];
/// let mut suite = SkillTaskSet::new("hello.v1");
/// suite.push(SkillTask::new("t1", "say hello").with_grader("greets"));
///
/// let report = SkillHarness::new(R, graders)
///     .with_trials(5)
///     .run(&suite)
///     .await?;
/// println!("{}", serde_json::to_string_pretty(&report)?);
/// # Ok(()) }
/// ```
pub struct SkillHarness<R: AgentRunner> {
    runner: R,
    graders: Vec<Box<dyn AsyncGrader>>,
    trials: usize,
    concurrency: usize,
    /// Score in `[0,1]` at which a per-(task, grader, trial) outcome is
    /// counted as a success when building [`ReliabilityReport`]s. Defaults
    /// to `1.0` (binary pass/fail). Lower this to accept partial credit
    /// from graders that use [`GraderOutcome::partial`].
    pass_threshold: f64,
}

impl<R: AgentRunner> SkillHarness<R> {
    /// Build a harness with a runner and a fixed grader registry.
    pub fn new(runner: R, graders: Vec<Box<dyn AsyncGrader>>) -> Self {
        Self {
            runner,
            graders,
            trials: 1,
            concurrency: 1,
            pass_threshold: 1.0,
        }
    }

    /// Number of trials per task. Must be `>= 1`. Defaults to `1`.
    #[must_use]
    pub fn with_trials(mut self, trials: usize) -> Self {
        self.trials = trials.max(1);
        self
    }

    /// Maximum concurrent `(task, trial)` runs. Defaults to `1`.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Pass threshold for reliability aggregation. Defaults to `1.0`.
    #[must_use]
    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    /// Configured trial count.
    #[must_use]
    pub fn trials(&self) -> usize {
        self.trials
    }

    /// Execute the suite end-to-end.
    ///
    /// Returns `Err(Error::Config)` if a task references a grader id that
    /// is not in the registry, or if `tasks` is empty. Errors from the
    /// runner are propagated.
    #[instrument(skip_all, fields(suite = %tasks.id, n_tasks = tasks.tasks.len(), trials = self.trials))]
    pub async fn run(&self, tasks: &SkillTaskSet) -> Result<SkillEvalReport> {
        if tasks.is_empty() {
            return Err(Error::Config("skill task set is empty".into()));
        }

        let registry: BTreeMap<&str, &dyn AsyncGrader> =
            self.graders.iter().map(|g| (g.id(), g.as_ref())).collect();
        if registry.len() != self.graders.len() {
            return Err(Error::Config(
                "duplicate grader id in registry; ids must be unique".into(),
            ));
        }

        // Validate that every grader id referenced by every task resolves.
        let mut referenced: BTreeSet<&str> = BTreeSet::new();
        for task in &tasks.tasks {
            for gid in &task.graders {
                if !registry.contains_key(gid.as_str()) {
                    return Err(Error::Config(format!(
                        "task {:?} references unknown grader {:?}",
                        task.id, gid
                    )));
                }
                referenced.insert(gid.as_str());
            }
        }

        // Build the flat work matrix `(task_index, trial_index)`.
        let work: Vec<(usize, usize)> = (0..tasks.tasks.len())
            .flat_map(|ti| (0..self.trials).map(move |tr| (ti, tr)))
            .collect();

        let registry_ref = &registry;
        let outcomes_stream = stream::iter(work.into_iter().map(move |(ti, tr)| {
            let task = tasks.tasks.get(ti);
            async move {
                let Some(task) = task else {
                    return Err(Error::Config(format!("task index {ti} out of bounds")));
                };
                debug!(task_id = %task.id, trial = tr, "running trial");
                let transcript = self.runner.run(task, tr).await?;
                let mut per_grader: BTreeMap<String, GraderOutcome> = BTreeMap::new();
                for gid in &task.graders {
                    let Some(grader) = registry_ref.get(gid.as_str()) else {
                        return Err(Error::Config(format!(
                            "grader {gid:?} disappeared between validation and run"
                        )));
                    };
                    let outcome = grader.grade(task, &transcript).await;
                    per_grader.insert(gid.clone(), outcome);
                }
                Ok::<_, Error>(TrialRow {
                    task_id: task.id.clone(),
                    trial: tr,
                    transcript,
                    outcomes: per_grader,
                })
            }
        }))
        .buffer_unordered(self.concurrency)
        .collect::<Vec<_>>()
        .await;

        let mut rows: Vec<TrialRow> = Vec::with_capacity(outcomes_stream.len());
        for row in outcomes_stream {
            rows.push(row?);
        }
        rows.sort_by(|a, b| {
            a.trial
                .cmp(&b.trial)
                .then_with(|| a.task_id.cmp(&b.task_id))
        });

        // For each grader id referenced by any task, build one MetricReport
        // per trial whose per-query entries are the tasks that include that
        // grader. Tasks that don't reference the grader are simply omitted.
        let mut per_grader_reports: BTreeMap<String, Vec<MetricReport>> = BTreeMap::new();
        for trial in 0..self.trials {
            let trial_rows: Vec<&TrialRow> = rows.iter().filter(|r| r.trial == trial).collect();
            for gid in &referenced {
                let pairs: Vec<(String, f64)> = trial_rows
                    .iter()
                    .filter_map(|row| {
                        row.outcomes
                            .get(*gid)
                            .map(|o| (row.task_id.clone(), o.score))
                    })
                    .collect();
                if pairs.is_empty() {
                    continue;
                }
                let report = MetricReport::from_per_query((*gid).to_string(), pairs);
                per_grader_reports
                    .entry((*gid).to_string())
                    .or_default()
                    .push(report);
            }
        }

        // Build per-grader reliability reports. Skip graders whose trial
        // count or task coverage prevents a meaningful pass^k estimate.
        let mut reliability: BTreeMap<String, ReliabilityReport> = BTreeMap::new();
        for (gid, trial_reports) in &per_grader_reports {
            if trial_reports.len() < self.trials {
                warn!(
                    grader = %gid,
                    have = trial_reports.len(),
                    want = self.trials,
                    "skipping reliability aggregation: not all trials produced this grader"
                );
                continue;
            }
            // Reliability requires equal task coverage per trial. If a task
            // appeared in trial A but not trial B, skip aggregation rather
            // than returning a misleading score.
            let first_task_set: BTreeSet<&str> = trial_reports
                .first()
                .map(|r| r.per_query.iter().map(|(q, _)| q.as_str()).collect())
                .unwrap_or_default();
            let consistent = trial_reports.iter().all(|r| {
                let set: BTreeSet<&str> = r.per_query.iter().map(|(q, _)| q.as_str()).collect();
                set == first_task_set
            });
            if !consistent {
                warn!(grader = %gid, "skipping reliability: task coverage varies across trials");
                continue;
            }
            match ReliabilityReport::from_metric_reports(
                gid.clone(),
                self.pass_threshold,
                self.trials,
                trial_reports,
            ) {
                Ok(r) => {
                    reliability.insert(gid.clone(), r);
                }
                Err(err) => {
                    warn!(grader = %gid, %err, "reliability aggregation failed");
                }
            }
        }

        Ok(SkillEvalReport {
            suite_id: tasks.id.clone(),
            n_tasks: tasks.tasks.len(),
            trials: self.trials,
            pass_threshold: self.pass_threshold,
            per_grader: per_grader_reports,
            reliability,
            trials_log: rows,
        })
    }
}

/// One captured `(task, trial)` row with the grader outcomes that fired
/// against it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRow {
    /// Task id.
    pub task_id: String,
    /// 0-based trial index.
    pub trial: usize,
    /// Captured transcript for the trial.
    pub transcript: Transcript,
    /// Grader outcomes keyed by grader id, restricted to the graders this
    /// task references.
    pub outcomes: BTreeMap<String, GraderOutcome>,
}

/// Aggregate output of a [`SkillHarness::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillEvalReport {
    /// Suite identifier copied from the input [`SkillTaskSet::id`].
    pub suite_id: String,
    /// Number of tasks evaluated.
    pub n_tasks: usize,
    /// Number of trials per task.
    pub trials: usize,
    /// Pass threshold used for reliability aggregation.
    pub pass_threshold: f64,
    /// Per-grader, per-trial metric reports (mean, percentiles, per-task).
    pub per_grader: BTreeMap<String, Vec<MetricReport>>,
    /// Per-grader reliability reports (pass rate, pass@k, pass^k). Empty
    /// for graders whose trial coverage was inconsistent — see the warning
    /// logs.
    pub reliability: BTreeMap<String, ReliabilityReport>,
    /// Raw per-trial rows for transcript inspection. Anthropic's roadmap:
    /// "read the transcripts."
    pub trials_log: Vec<TrialRow>,
}

impl SkillEvalReport {
    /// Convenience accessor returning the mean pass rate of `grader_id`,
    /// or `None` if the grader did not appear.
    #[must_use]
    pub fn mean_pass_rate(&self, grader_id: &str) -> Option<f64> {
        self.reliability.get(grader_id).map(|r| r.mean_pass_rate)
    }
}
