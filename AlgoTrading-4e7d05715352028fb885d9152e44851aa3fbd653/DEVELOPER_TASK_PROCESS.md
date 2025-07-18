# Developer Task Process Guide - Agent Version

## ABSOLUTE RULES (ZERO TOLERANCE)

1. **NEVER CODE WITHOUT CONTEXT**: Always examine existing codebase first
2. **NEVER HALLUCINATE**: If you don't know something, use tools or ask
3. **NEVER ASSUME**: Verify every technical detail before using it
4. **NEVER BREAK EXISTING CODE**: Test before and after changes
5. **NEVER SUBMIT INCOMPLETE WORK**: Full implementation only

## MANDATORY EXECUTION PROTOCOL

### STEP 1: CONTEXT GATHERING (REQUIRED PROOF)
**You MUST provide evidence of understanding:**

```
REQUIRED OUTPUT:
□ "Project structure analysis: [Describe key directories/files]"
□ "Existing patterns found: [List 2-3 specific patterns with examples]"
□ "Dependencies identified: [List files that will be affected]"
□ "Current functionality verified: [Describe what you tested]"
```

**Tools you MUST use when available:**
- Codebase analysis tools (examine structure, find patterns)
- File system access (verify file existence, read contents)
- Context7 MCP (get latest documentation when stuck)
- Web search (research best practices)

**ENFORCEMENT**: If you cannot provide the required output above, you MUST stop and ask for help.

### STEP 2: IMPACT VERIFICATION (REQUIRED PROOF)
**You MUST provide evidence of impact analysis:**

```
REQUIRED OUTPUT:
□ "Files to modify: [Exact list with reasons]"
□ "Files that depend on changes: [Exact list]"
□ "Breaking change risk assessment: [HIGH/MEDIUM/LOW with explanation]"
□ "Mitigation plan: [Specific steps if HIGH risk]"
```

**ENFORCEMENT**: If risk is HIGH and you have no mitigation plan, you MUST stop and ask questions.

### STEP 3: IMPLEMENTATION (STRICT STANDARDS)
**Follow this exact sequence:**

```
1. Match existing code style exactly (provide example)
2. Implement complete error handling (list all error cases)
3. Add input validation (specify validation rules)
4. Include comprehensive logging (show logging strategy)
5. Write complete implementation (no TODOs/placeholders)
```

**VERIFICATION LOOP**: After each major change, state:
- "Change made: [What you did]"
- "Tested: [How you verified it works]"
- "Impact check: [Confirmed no regressions]"

### STEP 4: FINAL VALIDATION (MANDATORY CHECKLIST)
**You MUST confirm each item:**

```
□ All requirements implemented (list requirements met)
□ All error scenarios handled (list error cases covered)
□ Existing functionality preserved (describe tests performed)
□ Code follows project patterns (provide pattern examples)
□ Performance acceptable (specify performance criteria)
```

**ENFORCEMENT**: Task is NOT complete until ALL checkboxes are verified.

## CONTEXT GATHERING RULES

### When to Use Context7 MCP Tool:
- **ALWAYS** when you need latest library documentation
- **ALWAYS** when you're unsure about API usage
- **ALWAYS** when implementing with unfamiliar libraries
- **ALWAYS** when you encounter errors or unexpected behavior

### Information Verification Protocol:
**Before stating ANY technical fact, you MUST:**
```
□ Source: [Where did this information come from?]
□ Verified: [Which tool confirmed this?]
□ Evidence: [What proof do you have?]
```

### Prohibited Statements:
- "I assume this file exists" → USE: "I verified this file exists using [tool]"
- "This function probably takes" → USE: "I confirmed this function signature using [tool]"
- "The API likely supports" → USE: "I verified the API supports [feature] using Context7"

## ENFORCEMENT MECHANISMS

### Mandatory Stops (You MUST halt and ask):
- **Cannot find required files or information**
- **Unclear about project structure or patterns**
- **Risk assessment shows HIGH risk without mitigation**
- **Encountering errors you cannot resolve**
- **Need to make assumptions about requirements**

### Question Format (Use this exact template):
```
**MANDATORY STOP - CLARIFICATION NEEDED**

**Verified so far:**
- [List what you've confirmed using tools]
- [List patterns you've identified]
- [List dependencies you've mapped]

**Unable to verify:**
- [Specific unknowns preventing progress]
- [Tools attempted and results]

**Need clarification on:**
- [Specific questions with context]

**Risk if I proceed without clarification:**
- [What might break or go wrong]
```

### Progressive Enforcement:
**Phase 1**: If you skip context gathering → MANDATORY STOP
**Phase 2**: If you cannot prove impact analysis → MANDATORY STOP  
**Phase 3**: If you write incomplete code → MANDATORY STOP
**Phase 4**: If validation fails → MANDATORY STOP

## ANTI-HALLUCINATION SYSTEM

### Real-time Verification:
**Before every technical statement, ask yourself:**
1. "Did I verify this using available tools?"
2. "Do I have evidence for this claim?"
3. "Am I guessing or do I know this?"

### Evidence Requirements:
- **File paths**: Must verify existence before referencing
- **Function signatures**: Must confirm using tools or documentation
- **API capabilities**: Must verify using Context7 or official docs
- **Code patterns**: Must identify in existing codebase
- **Dependencies**: Must confirm through actual code analysis

### Confidence Levels (Required):
- **VERIFIED**: "I confirmed using [tool] that [fact]"
- **DOCUMENTED**: "Context7 documentation shows [fact]"
- **OBSERVED**: "I found this pattern in [specific file]: [example]"
- **UNKNOWN**: "I cannot verify [fact] - need clarification"

## RECOVERY PROCEDURES

### If You Make a Mistake:
1. **Acknowledge immediately**: "I made an error in [specific area]"
2. **Stop current work**: Don't continue with incorrect assumptions
3. **Revert changes**: Return to last verified working state
4. **Re-gather context**: Use tools to get correct information
5. **Ask for help**: Use the mandatory question format

### If You're Stuck:
1. **Use Context7 MCP**: Get latest documentation for libraries/frameworks
2. **Use available tools**: Exhaust all information gathering options
3. **Ask specific questions**: Use the mandatory format above
4. **Don't guess**: Better to ask than to implement incorrectly

## SUCCESS CRITERIA

**Task completed successfully when:**
- ✅ **Context proven**: You can describe project structure and patterns
- ✅ **Impact verified**: You can list all affected files and risks
- ✅ **Implementation complete**: No TODOs, all requirements met
- ✅ **Validation passed**: All tests pass, no regressions
- ✅ **Evidence provided**: You can prove each claim with sources

**FINAL ENFORCEMENT**: Before submitting work, you MUST provide:
```
**COMPLETION VERIFICATION**
□ Context gathered using: [list tools used]
□ Patterns followed: [specific examples from codebase]
□ Testing performed: [describe verification steps]
□ Documentation consulted: [Context7 or other sources]
□ Risks mitigated: [how you prevented breaks]
```

**Remember**: This is not a suggestion - it's a mandatory protocol. Following these steps prevents bugs, reduces rework, and ensures production-ready code. When in doubt, gather more context or ask questions.