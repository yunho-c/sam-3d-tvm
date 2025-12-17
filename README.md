# sam-3d-tvm

SAM 3D with TVM acceleration.

## Quick Start

### Using Pixi (Recommended)

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies and create environment
pixi install

# Run tests
pixi run test

# Lint code
pixi run lint

# Format code
pixi run format
```

### Using Hatch

```bash
# Install hatch if you haven't already
pip install hatch

# Run tests
hatch run test

# Lint code
hatch run lint

# Format code
hatch run format
```

## Development

### Project Structure

```
sam-3d-tvm/
├── src/
│   └── sam_3d_tvm/       # Main package
│       ├── __init__.py
│       └── py.typed      # PEP 561 marker
├── tests/                # Test files
├── pyproject.toml        # Project configuration
└── README.md
```

### Code Quality

This project uses:
- **ruff** for linting and formatting (120 char line length)
- **mypy** for static type checking
- **pytest** for testing

```bash
# Run all checks
pixi run lint
pixi run typecheck
pixi run test
```

## License

MIT
