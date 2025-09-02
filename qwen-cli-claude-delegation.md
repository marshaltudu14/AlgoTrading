# Senior Developer Workflow: Delegating to Qwen CLI

As the senior developer, break down high-level tasks into small, specific tasks and delegate the bulk code writing to Qwen CLI. Review and approve each implementation before moving to the next task.

## Delegation Strategy

### Task Decomposition

1. **Receive high-level requirement** from user
2. **Break down into small, atomic tasks** (each should be completable in one Qwen call)
3. **Define clear acceptance criteria** for each task
4. **Assign tasks to Qwen one by one**
5. **Review Qwen's implementation** before proceeding
6. **Iterate or approve** and move to next task

### Using Qwen CLI for Task Implementation

**Basic delegation syntax:**

```bash
qwen -p "Task: [specific task description]

Context: [any relevant context or existing code]

Requirements:
- [specific requirement 1]
- [specific requirement 2]
- [etc.]

Please implement this following [coding standards/patterns if applicable]"
```

## Example Task Breakdown

**High-level requirement:** "Build a user authentication system"

**Senior Dev (Claude) breaks it down:**

### Task 1: Database Schema

```bash
qwen -p "Task: Create user authentication database schema

Requirements:
- Users table with id, email, password_hash, created_at, updated_at
- Use proper data types and constraints
- Include indexes for email lookups
- Write the SQL migration file

Follow standard naming conventions and include proper foreign key constraints if needed."
```

### Task 2: User Model

```bash
qwen -p "Task: Create User model class

Requirements:
- User class with email and password fields
- Password hashing using bcrypt
- Email validation
- Class methods for authentication
- Follow existing project patterns in @models/

Context: Here's the existing base model structure: [provide base model if exists]"
```

### Task 3: Registration Endpoint

```bash
qwen -p "Task: Implement user registration API endpoint

Requirements:
- POST /api/register endpoint
- Validate email format and password strength
- Hash password before storing
- Return JWT token on success
- Handle duplicate email errors
- Follow existing API patterns in @routes/

Context: Here's the existing route structure: [provide example route]"
```

## Review Process

After each Qwen task completion:

1. **Code Review Checklist:**

   - Does it meet all specified requirements?
   - Follows project coding standards?
   - Handles edge cases appropriately?
   - Includes proper error handling?
   - Security considerations addressed?

2. **Approval Actions:**

   - ✅ **APPROVE**: "Implementation looks good, proceeding to next task"
   - 🔄 **ITERATE**: Give Qwen specific feedback for improvements
   - ✏️ **MANUAL FIX**: Make small adjustments yourself if faster

3. **Integration Testing:**
   - Test the implemented feature
   - Ensure it works with existing code
   - Verify all requirements are met

## Task Assignment Best Practices

### Make Tasks Atomic

❌ **Bad**: "Build the entire auth system"
✅ **Good**: "Create the login validation function"

### Provide Clear Context

```bash
qwen -p "Task: Add error handling to payment processing

Context: We have an existing payment function at @src/payments/processor.js
The function currently doesn't handle network timeouts or API failures

Requirements:
- Add try-catch blocks
- Handle network timeouts (30s max)
- Handle API error responses
- Log errors appropriately
- Return user-friendly error messages"
```

### Specify Output Format

```bash
qwen -p "Task: Create API documentation for user endpoints

Requirements:
- Document all 4 user CRUD endpoints
- Include request/response examples
- Use OpenAPI 3.0 format
- Include error response codes
- Save as docs/api/users.yaml"
```

## Workflow Commands

**Assign new task to Qwen:**

```bash
qwen -p "[detailed task specification]"
```

**Get Qwen to fix issues:**

```bash
qwen -p "The previous implementation has these issues: [list issues]
Please fix these problems: [specific fixes needed]
Here's the current code: [provide code if needed]"
```

**Have Qwen refactor/improve:**

```bash
qwen -p "Refactor this code to improve [specific aspect]:
[provide current code]
Focus on: [specific improvements needed]"
```

## Senior Dev Responsibilities

- **Task Planning**: Break down requirements into manageable chunks
- **Quality Gate**: Review every implementation before approval
- **Architecture Decisions**: Make high-level design choices
- **Integration**: Ensure all pieces work together
- **Testing Strategy**: Define what needs to be tested
- **Code Standards**: Enforce consistent patterns and practices
- **Performance Review**: Identify bottlenecks and optimization opportunities

## Notes

- Each task should be completable by Qwen in a single response
- Always review before moving to the next task
- Provide context about existing codebase patterns
- Be specific about requirements and acceptance criteria
- Use Qwen for bulk implementation, Claude for oversight and integration
