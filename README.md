# PTStudio: Simple 3D scene editor and renderer written in C++17
PTStudio is a light-weight, modular, and extensible 3D scene editor and renderer written in C++17. It is my personal hobby project to learn modern C++, OpenGL, CUDA, Vulkan, and other graphics-related technologies. It is also a playground for me to experiment with new ideas and techniques.

## Disclaimer
This project is still in its early stage and might be quite buggy. I am still **actively** working on the basic functionalities. See the section below for the current roadmap and feature overview of this project. As the project matures, I will add more documentation, examples, demos, and tests. The source code structure is still undergoing major changes, so I may not be able to accept pull requests at this time. However, I am always open to suggestions and minor fixes. Please feel free to open an issue if you have any questions or suggestions.

## Screenshots and Videos
![Screenshot](docs/readme_assets/scene_editing.png)

https://github.com/dw218192/PTStudio/assets/31294154/6c956cb8-3d53-4e00-8419-cb108d74fecd

## Build Instructions

### Prerequisites
- CMake 3.19
- C++17 compiler
- CUDA, Vulkan, OpenGL

### Dependencies
- make sure to recursively clone the submodules using
```bash
git clone --recursive [repo url]
```

## Features (As of 2023-11-05)
- [ ] Application framework
    - [x] GLFW window, input handling
    - [x] ImGui integration, basic UI
    - [x] Simple Reflection system
    - [x] Renderer Interface
    - [x] Serialization/Deserialization Interface
    - [ ] Primitive Scene Data Types
        - [x] Light
        - [x] Camera
        - [x] Mesh
        - [x] Material
        - [x] Texture
        - [ ] ...
    - [ ] Asset/ Resource Management System
- [ ] Scene Editing
    - [x] Basic scene object manipulation, Gizmo
    - [ ] Scene object duplication
    - [x] Scene object creation, deletion
    - [x] Scene object selection, ray picking
    - [x] Multiple light support
    - [ ] Scene object hierarchy
    - [ ] Scene object component system
        - [ ] Scene object custom data
    - [ ] Undo/Redo System
    - [ ] OpenGL-based Editor Renderer
        - [x] Outline, Sprite rendering
        - [x] Grid rendering
        - [x] GLSL shader hot reload
        - [x] GLSL shader editor
        - [ ] GLSL shader save/load
    - [x] Json serialization
    - [ ] Mesh loading
        - [ ] Multi-threaded mesh loading
        - [ ] Texture loading
        - [x] .obj file loading
    - [ ] Renderer switching
        - [ ] Dynamic renderer DLL loading
- [ ] Renderers
    - [ ] CUDA Path Tracer
    - [ ] Vulkan hardware ray tracer
    - [ ] OpenGL deferred renderer
