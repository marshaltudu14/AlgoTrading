---
Status: Approved
Story:
  Sequence: 4.2
  Title: Integrate the Autonomous Stage into the Main Pipeline
  Description: As a developer, I want to modify the main `run_training.py` script to recognize and execute the new "Autonomous" stage so that it becomes the final step in the training pipeline.
Acceptance Criteria:
  1: The `run_training.py` script is modified to include a condition for `algorithm: "Autonomous"`.
  2: When this condition is met, it calls the `run_autonomous_stage` function from the new training module.
  3: The script can successfully run the full PPO -> MoE -> MAML -> Autonomous pipeline in sequence.
  4: The `config/training_sequence.yaml` is updated with a complete `stage_4_autonomous` section, including all necessary settings.
Tasks:
  - Task: Modify `config/training_sequence.yaml`
    Subtasks:
      - Subtask: Add `stage_4_autonomous` with `algorithm: "Autonomous"` and `prerequisites: "stage_3_maml"`
      - Subtask: Add `evolution_settings`, `nas_settings`, `world_model_settings`, `memory_settings`, `reasoning_settings` to `stage_4_autonomous`
  - Task: Modify `run_training.py`
    Subtasks:
      - Subtask: Locate the main training loop that iterates through stages
      - Subtask: Add an `elif` condition for `stage_config['algorithm'] == 'Autonomous'`
      - Subtask: Inside the `elif` block, import `run_autonomous_stage` from `src/training/autonomous_trainer.py`
      - Subtask: Call `run_autonomous_stage(stage_config)` to execute the autonomous stage
  - Task: Verify full pipeline execution
    Subtasks:
      - Subtask: Run `run_training.py` with a configuration that includes all four stages
      - Subtask: Confirm that all stages execute sequentially without errors
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: