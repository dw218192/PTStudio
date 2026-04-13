# Pending tickets — resume here

Stashed from session on 2026-04-13 for resumption on another machine.

Work already landed on `dev/rendering-next` as single squashed commit `f5f6679`:
- DepTrackedCache + RenderWorld versioning
- IShaderCompiler interface + EmbeddedCompiler + SlangCompiler (disk cache, mtime watcher, dep capture)
- Config-driven shader variants (schema + codegen)
- Cleanup: lowerCamelCase renames, boost::hash_combine (dropped hand-rolled Sha256), UNUSED macro, ShaderKey struct

## Pending

1. **`linux-tool-builds`** — prerequisite. Make C++ build-time tools buildable on Linux so we can stop smuggling `usdz_pack` from Windows CI to Emscripten CI. Verification via a temporary GitHub Actions workflow (CI-runner iteration; rejected Docker and WSL for conan cache bootstrap cost and env drift).
2. **`cpp-shader-compiler-tool`** — depends on #1. Replace Python `slangc.py` with a C++ CLI wrapping `IShaderCompiler::compile()`. Unblocks dropping `slangc` from native prebuild (descoped from `shader-variants-config`).

## Iteration pattern

For `linux-tool-builds`: orchestrator-driven. Worker dispatches make specific code changes (portable conanfile, CMake, profile, temp workflow yaml), orchestrator handles push + CI wait + log fetch between dispatches. Sub-agents summarize long CI logs to keep orchestrator context clean. Headless single-dispatch workers can't handle the CI wait loop within their timeout.

## Files

- `linux-tool-builds.md` — full ticket description + acceptance criteria
- `cpp-shader-compiler-tool.md` — full ticket description + acceptance criteria

Both tickets also exist in the project's ticket system (`_agent/tickets/`) — these markdown copies are the canonical source if the ticket system gets out of sync.
