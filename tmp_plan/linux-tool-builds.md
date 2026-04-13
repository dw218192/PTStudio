# linux-tool-builds

**Title:** Make C++ build-time tools buildable on Linux (kill Windows-artifact smuggling)

**Status:** todo

Enable C++ build-time tools (currently just `usdz_pack`; soon `pts_shaderc`) to build on Linux. Today Windows CI produces `usdz_pack` and Emscripten CI grabs the artifact — this entrenches a brittle cross-platform dependency and blocks adding more tools.

## Goal

Linux CI can produce the full set of C++ build-time tools from source. Emscripten CI consumes them from its own Linux build, not from Windows.

## Non-goals

- Full Linux runtime build (editor, renderers, tests) — out of scope. Only the **tool subset** matters here.
- Migrating existing CI workflows — that's a follow-up once the tool-build workflow is green.

## Scope

### Tool subset

Currently just `tools/conan/usdz_pack/` (`usdzPack.cpp`). A future ticket (`cpp-shader-compiler-tool`) adds `pts_shaderc`. Both must build on Linux.

### Work

1. **Audit `tools/conan/usdz_pack/conanfile.py` + `CMakeLists.txt`** for Windows-isms: hardcoded MSVC flags, Windows-only headers, path separators.
2. **Add/fix Linux conan profile** (`tools/conan/profiles/conan_profile_linux`?) covering compiler (gcc or clang), libc++/libstdc++, cppstd=17, shapes matching host+build profiles used today.
3. **Fix CMake** portability: `CMAKE_CXX_STANDARD`, avoid platform-specific targets, guard any Windowing flags.
4. **Update repokit tool-build paths** if they assume Windows layout.
5. **Document** the Linux tool-build invocation in `CLAUDE.md` or `tools/conan/README.md`.

## Verification: CI runner iteration

Local verification via Docker/WSL both have downsides (conan cache bootstrap, env drift). Use GitHub Actions as the iteration surface instead.

### Approach

1. Worker creates a **temporary workflow** scoped to this ticket's feature branch, e.g. `.github/workflows/linux-tool-build-smoke.yml`, that:
   - Runs on `ubuntu-latest`
   - Installs prereqs (`g++`, `cmake`, `ninja`, `python3`, `pip install conan`)
   - Runs `./repo build --platform linux-x64 --tool-only usdz_pack` (or equivalent — part of this ticket is figuring out the right invocation)
   - Caches `~/.conan2` via `actions/cache@v4` keyed on `conanfile.py + profile` so iterations don't re-download everything
   - Runs the produced binary against a smoke input to confirm it's usable
2. Orchestrator pushes the branch, watches CI, reads logs, dispatches next worker with a targeted change prompt. (Headless worker can't poll CI within its timeout budget.)
3. First runs will be slow (cold conan cache); subsequent runs hit the actions cache.
4. Before merging: **delete the temporary workflow file** unless we decide to keep it as a permanent Linux tool-build CI gate (probably yes, but that's a judgment call at merge time).

### Iteration budget

GitHub Actions minutes are the real cost. Keep the workflow:
- Fail-fast enabled
- Cache aggressively (conan cache, ninja object cache if practical)
- Only `ubuntu-latest` — don't matrix across distros/compilers in this ticket

Target: 10-20 iterations to land. Each iteration ~5-15 min (first is longer).

## Acceptance criteria

- `usdz_pack` builds from source on ubuntu-latest via the temporary GitHub Actions workflow end-to-end
- Workflow uses `actions/cache` for `~/.conan2` keyed on conanfile+profile so iterations don't re-download
- Produced `usdz_pack` binary runs and packages a test `.usdz` scene (smoke test in the workflow)
- Linux conan profile committed or existing profile patched; referenced by the tool-build path
- Windows build of `usdz_pack` still works unchanged (no regression)
- `CLAUDE.md` or `tools/conan/README.md` documents the Linux tool-build invocation
- Any Windows-specific code paths in `conanfile.py` / `CMakeLists.txt` are portable or explicitly platform-guarded
- Decision committed: workflow either promoted (kept with justified trigger scope) or removed before merge
- **Fail loud**: tool build failures surface as hard errors, no silent skips

## Risks

- **Conan package conflicts on Linux**: OpenUSD/TBB/etc. may have Linux-specific gotchas. Record surprises in progress notes so `pts_shaderc` avoids them later.
- **Actions cache invalidation**: conan cache key must include the profile and conanfile hash; otherwise caches go stale silently.
- **Workflow file drift**: if we promote the temporary workflow, make sure its trigger scope is right (on push to main? on PRs touching tool files? — avoid running it on every unrelated push).

## Out of scope

- Migrating the Emscripten CI workflow to consume Linux artifacts (follow-up).
- Porting the runtime (editor, renderers, tests) to Linux.
- The `pts_shaderc` tool itself — see `cpp-shader-compiler-tool` (depends on this ticket).
