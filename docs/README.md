# KG-Service Documentation

This directory contains comprehensive technical documentation for the Knowledge Graph Service.

## Documentation Structure

- **[index.md](index.md)**: Main landing page with overview and quick start
- **[architecture.md](architecture.md)**: Complete system architecture, data flow, and component interactions
- **[api.md](api.md)**: FastAPI server, endpoints, request/response models, and integration contracts
- **[extractors.md](extractors.md)**: Entity and relationship extraction engines (GLiNER and vLLM)
- **[pipeline.md](pipeline.md)**: Processing orchestration, chunk mapping, and workflow coordination
- **[storage.md](storage.md)**: Neo4j client, graph schema, and database operations
- **[configuration.md](configuration.md)**: Service configuration, environment variables, and external clients

## GitHub Pages Deployment

This documentation is designed for GitHub Pages with Jekyll.

### Setup

1. Push the `/docs` folder to your repository
2. Go to repository Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select branch `main` and folder `/docs`
5. Save

GitHub Pages will automatically build and deploy the documentation.

### Local Preview

To preview locally with Jekyll:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Visit http://localhost:4000

### Theme

The documentation uses the `jekyll-theme-cayman` theme (configured in `_config.yml`).

## Documentation Standards

### Writing Style
- **Technical and precise**: Written for senior programmers
- **No code snippets**: Explains interfaces and communication patterns without code examples
- **Conceptual focus**: Emphasizes architecture, data flow, and design decisions
- **Implementation details**: Describes how components work internally

### Coverage
Each module documentation includes:
- Purpose and responsibility
- Architecture and dependencies
- Key interfaces and methods
- Data flow and processing steps
- Error handling strategies
- Performance characteristics
- Integration points

## Updating Documentation

When making code changes:

1. Update relevant documentation files
2. Maintain consistency across linked pages
3. Ensure all cross-references are valid
4. Test navigation between pages
5. Commit documentation changes with code changes

## Navigation

The documentation is interconnected with "Next" links at the bottom of each page:

```
index.md → architecture.md → api.md → extractors.md → pipeline.md → storage.md → configuration.md
```

## Contact

For questions or improvements, open an issue in the repository.
