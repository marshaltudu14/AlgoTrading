---
Status: Approved
Story:
  Sequence: 2.2
  Title: Implement the External Memory Module
  Description: As a developer, I want to create an `ExternalMemory` module where the agent can store and retrieve significant past events so that it can learn from specific historical experiences.
Acceptance Criteria:
  1: A new directory `src/memory/` is created.
  2: A new file `src/memory/episodic_memory.py` is created within it.
  3: The file contains an `ExternalMemory` class.
  4: The class has a `store(event_embedding, outcome)` method to save a vector representation of an event.
  5: The class has a `retrieve(current_state_embedding)` method that returns the most similar past events from memory.
  6: The initial implementation can use a simple vector similarity search library (e.g., Faiss) or even NumPy for the backend, ensuring efficient querying by the `AutonomousAgent`.
  7: A unit test is created to verify that memories can be stored and retrieved correctly.
Tasks:
  - Task: Create `src/memory/` directory and `episodic_memory.py`
    Subtasks:
      - Subtask: Create the directory structure
      - Subtask: Define `ExternalMemory` class
  - Task: Implement `store` method
    Subtasks:
      - Subtask: Define `store(event_embedding, outcome)` method
      - Subtask: Choose a data structure for storing memories (e.g., list of tuples, dictionary)
      - Subtask: Implement logic to add new memories to the store
  - Task: Implement `retrieve` method
    Subtasks:
      - Subtask: Define `retrieve(current_state_embedding)` method
      - Subtask: Implement a similarity search (e.g., cosine similarity with NumPy, or integrate a basic Faiss index)
      - Subtask: Return top-K most similar memories
  - Task: Create unit tests for `ExternalMemory`
    Subtasks:
      - Subtask: Create `tests/test_episodic_memory.py`
      - Subtask: Write test case to verify `store` functionality
      - Subtask: Write test case to verify `retrieve` functionality with dummy embeddings
      - Subtask: Test edge cases (empty memory, no similar memories)
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: