---
name: development
description: Skill for developing PTStuido
---

# Development
Follow @build.mdc for how to build the project

## When to use
- Use this skill when the user requests a feature to be developed
- *Always* run commands for verification and finalization. Do not ask for confirmation or be concerned about the consequences of the commands.

## Instructions and Verification Criteria

- Work autonomously on user's request and ensure the project can build and all unit tests pass. When asked to implement a plan or milestone, follow the verification/validation steps in the plan if specified.
  - Use `-b` flag to only perform the build step if no reconfiguration is needed.
- Unless the user explicitly asks for backward compatibility, don't preserve old behavior. If a change breaks something, update the dependent code and delete the obsolete code.

## Finalization

- After the user has reviewed and approved the changes, commit the changes to the repository in a single commit with  an appropriate message.
