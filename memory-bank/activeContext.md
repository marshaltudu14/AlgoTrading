# Active Context: RL Algorithmic Trading Agent

## Current Focus

- **Initial Documentation:** Establishing the foundational Memory Bank files (`projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`, `activeContext.md`, `progress.md`).
- **Project Understanding:** Gaining initial context about the project structure, technologies, and apparent goals based on existing files and directory layout.

## Recent Changes

- Created the core Memory Bank markdown files based on analysis of the project directory.
- Read `requirements.txt` to identify dependencies.

## Next Steps (Immediate)

1.  Create the final core Memory Bank file: `progress.md`.
2.  Await further instructions or tasks from the user now that the initial context is documented.

## Active Decisions & Considerations

- The initial documentation is based on inferences from the file structure and naming conventions. It may need refinement as more specific details about the project's implementation and goals become available.
- The role of `src/signals.py` and the exact implementation details within `src/rl_environment.py` and `src/data_handler.py` are key areas for future exploration.
- The configuration details in `dynamic_config.json` will be important for understanding how the system is parameterized.

## Key Patterns & Preferences (Observed/Inferred)

- **Modularity:** Code seems organized into distinct modules within the `src/` directory (config, data handling, RL environment, Fyers auth).
- **Standard Libraries:** Reliance on well-established libraries for RL (`stable-baselines3`, `gymnasium`), data science (`pandas`, `numpy`, `scikit-learn`), and API interaction (`fyers-apiv3`).
- **Logging:** Emphasis on logging for different components (RL training, Fyers API).
- **Configuration Driven:** Use of a JSON file (`dynamic_config.json`) for managing parameters.

## Learnings & Insights

- The project is a relatively standard implementation of an RL trading agent pipeline.
- Significant work has already been done, evidenced by the processed data, trained models, and logging infrastructure.
- The use of PPO suggests a focus on continuous action spaces or a complex discrete action space suitable for trading.
