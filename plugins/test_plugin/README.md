# Test Plugin

A minimal test plugin demonstrating the PTStudio plugin system.

## Features

- Implements the basic plugin API (on_load/on_unload)
- Uses C ABI for cross-compiler compatibility
- Simple console logging for demonstration

## Building

This plugin is built automatically when you build the main project:

```powershell
pts.cmd build
```

## Packaging

After building, package the plugin to the bin/plugins directory:

```powershell
pts.cmd package
```

## Testing

The plugin will be automatically discovered by the PluginManager when the editor starts if it's in the `bin/plugins` directory.

