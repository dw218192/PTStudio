---
name: development
description: Skill for developing PTStuido
---

# Development
Follow @build.mdc for how to build the project

## When to use
- Use this skill when the user requests a feature to be developed

## Instructions
- Work autonomously on user's request and ensure the project can build and all unit tests pass. When asked to implement a plan or milestone, follow the verification/validation steps in the plan.
    - Use `-b` flag to only perform the build step if no reconfiguration is needed
- Unless the user explicitly asks for backward compatibility, don’t preserve old behavior. If a change breaks something, update the dependent code and delete the obsolete code.