//! Run a deterministic skill-eval suite from JSONL.
//!
//! ```sh
//! cargo run --example skills_basic --features skills
//! ```

use std::future::Future;

use rig_retrieval_evals::Result;
use rig_retrieval_evals::skills::{
    AgentRunner, AsyncGrader, ContainsGrader, SkillHarness, SkillTask, SkillTaskSet, ToolCall,
    ToolCallGrader, Transcript, TriggerGrader,
};

struct DemoRunner;

impl AgentRunner for DemoRunner {
    fn run(
        &self,
        task: &SkillTask,
        _trial: usize,
    ) -> impl Future<Output = Result<Transcript>> + Send {
        let prompt = task.prompt.clone();
        async move {
            if prompt.contains("search") {
                Ok(Transcript {
                    prompt,
                    final_output: "searched the Rig docs and found the relevant section".into(),
                    tool_calls: vec![ToolCall::new("search")],
                    skill_invoked: Some("search".into()),
                    ..Default::default()
                })
            } else {
                Ok(Transcript {
                    prompt,
                    final_output: "hello operator".into(),
                    ..Default::default()
                })
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let suite = SkillTaskSet::from_jsonl_str(
        "skills-basic.v1",
        r#"
        {"id":"greet","prompt":"say hello to the operator","graders":["greets"]}
        {"id":"search-docs","prompt":"search for the Rig docs","should_trigger":"search","graders":["used_search","triggered_search"]}
        "#,
    )?;

    let graders: Vec<Box<dyn AsyncGrader>> = vec![
        Box::new(ContainsGrader::present("greets", "hello")),
        Box::new(ToolCallGrader::at_least_once("used_search", "search")),
        Box::new(TriggerGrader::new("triggered_search")),
    ];

    let report = SkillHarness::new(DemoRunner, graders)
        .with_trials(2)
        .with_concurrency(2)
        .run(&suite)
        .await?;

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
