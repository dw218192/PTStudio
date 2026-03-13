# C++ Naming Conventions

- **Types**: UpperCamelCase (e.g. `RenderWorld`, `BufferReadback`)
- **Functions/methods**: snake_case (e.g. `tick_until_settled`, `try_read_u32`)
- **Variables/members**: snake_case (e.g. `pixel_value`, `m_buffer`)
- **Private members**: m_prefix + snake_case (e.g. `m_needs_map`, `m_device`)
- **constexpr / const constants**: k_snake_case (e.g. `k_orbit_speed`, `k_max_pitch`, `k_uniform_align`)
- **Namespaces**: snake_case (e.g. `pts::webgpu`, `pts::rendering`)
- **Macros / preprocessor**: UPPER_SNAKE_CASE
