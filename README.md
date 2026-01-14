# PTStudio: Simple 3D scene editor and renderer written in C++17
PTStudio is a light-weight, modular, and extensible 3D scene editor and renderer written in C++17. It is my personal hobby project to learn modern C++, OpenGL, Vulkan, and other graphics-related technologies. It is also a playground for me to experiment with new ideas and techniques.

## Screenshots and Demos
https://github.com/dw218192/PTStudio/assets/31294154/4116308d-ee31-4d20-a449-f130d11fe253

![Screenshot](docs/readme_assets/scene_editing.png)
![Screenshot](docs/readme_assets/cornell_editing.png)
![Screenshot](docs/readme_assets/cornell.png)

## Repository Tooling
- This project uses a hermetic tooling initialization process where all the tools are pulled in as Python packages, and python is used to perform various tasks like building, formatting, etc.
- To see the available tool commands, run `pts.cmd --help`.

## Build & Test Instructions
- Run the repository tooling entrypoint `pts.cmd build` to bootstrap the build system (if needed) and build the project.
- Run the repository tooling entrypoint `pts.cmd test` to run the unit tests.

### Prerequisites
- C++ Compiler Toolchain (GCC, Clang, MSVC, etc.)
- GPU driver with Vulkan and OpenGL support

### Dependencies
- Make sure to clone the repository with all submodules:
```bash
git clone --recursive [repo url]
```
- If you've already cloned without `--recursive`, you can fetch submodules afterward with:
```bash
git submodule update --init --recursive
```

## TODOs
### Feature
    - [ ] Rework the crazy reflection system
    - [ ] Better scene graph (consider OpenUSD)
### Code Improvement
    - [ ] Reduce the cumbersome use of `tl::expected` and use exceptions in subsystem internals
    - [ ] Renderers should be DLLs (this might be moot if moving to OpenUSD)
    - [ ] Editing Improvements
        - [ ] put mesh loading in a separate thread
        - [ ] adaptive grid resizing
        - [ ] Ctrl+S to save scene (first time a file dialog will pop up, then subsequent saves will save to the same file)
        - [ ] Ctrl+D and Ctrl+C to duplicate selected objects
        - [ ] Ctrl+Z and Ctrl+Y to undo/redo
        - [x] camera should move faster the further away it is from the center