# Documentation

This directory contains comprehensive documentation for the Phosphorylation Site Prediction Framework.

## Documentation Structure

### Getting Started
- **[getting_started.md](getting_started.md)** - Installation, setup, and basic usage guide

### Configuration
- **[configuration.md](configuration.md)** - Detailed configuration options and examples

### Models
- **[models.md](models.md)** - In-depth guide to XGBoost, Transformer, and Ensemble models

### Advanced Usage
- **[advanced_usage.md](advanced_usage.md)** - Advanced features, custom implementations, and optimization

### API Reference
- **[api_reference.md](api_reference.md)** - Complete API documentation with examples

## Quick Navigation

### New Users
1. Start with [getting_started.md](getting_started.md) for installation and basic usage
2. Review [configuration.md](configuration.md) to understand parameter tuning
3. Read [models.md](models.md) to choose the right model for your use case

### Experienced Users
1. Check [advanced_usage.md](advanced_usage.md) for custom implementations
2. Reference [api_reference.md](api_reference.md) for programmatic usage
3. Use [configuration.md](configuration.md) for detailed parameter optimization

### Developers
1. Start with [api_reference.md](api_reference.md) for class and method documentation
2. Review [advanced_usage.md](advanced_usage.md) for extension patterns
3. Check source code in `src/` for implementation details

## Documentation Conventions

### Code Examples
All code examples are tested and should work with the current version of the framework.

### Configuration Examples
Configuration examples use YAML format and include comments explaining each parameter.

### File Paths
All file paths in examples are relative to the project root directory.

### Command Line Examples
Command line examples assume you're in the project root directory and have installed dependencies.

## Additional Resources

### Project Structure
```
phosphorylation_prediction/
├── docs/                   # This documentation
├── src/                    # Source code
├── config/                 # Configuration files
├── scripts/               # Command-line scripts
├── tests/                 # Test files
└── experiments/           # Experiment outputs
```

### Key Concepts

**Models**
- XGBoost: Feature-based gradient boosting
- Transformer: Deep learning with protein language models
- Ensemble: Combinations of multiple models

**Features**
- AAC: Amino acid composition
- DPC: Dipeptide composition
- TPC: Tripeptide composition
- Binary encoding: One-hot encoded sequences
- Physicochemical: Protein property features

**Experiments**
- Single model: Train and evaluate one model
- Ensemble: Train and evaluate ensemble models
- Cross-validation: K-fold validation with proper splitting

### Common Workflows

1. **Quick Start**: XGBoost model with default configuration
2. **Research**: Cross-validation with ensemble models
3. **Production**: Optimized single model with custom features
4. **Comparison**: Multiple experiments with statistical analysis

## Contributing to Documentation

### Guidelines
- Use clear, concise language
- Include working code examples
- Explain both "what" and "why"
- Keep examples up-to-date with code changes

### Adding New Documentation
1. Create markdown files in the `docs/` directory
2. Follow existing format and style
3. Update this README to include new files
4. Test all code examples

### Reporting Issues
- Documentation bugs: Open GitHub issue
- Missing information: Submit feature request
- Unclear explanations: Suggest improvements

## Help and Support

### Getting Help
1. Check this documentation first
2. Search existing GitHub issues
3. Open new issue with specific details
4. Contact development team

### Community
- GitHub Discussions for questions
- Issues for bug reports
- Pull requests for contributions

## Version Information

This documentation is for version 1.0.0 of the Phosphorylation Site Prediction Framework.

For version-specific changes, see the project CHANGELOG.md.