MOMENTS sample fixture for loader smoke tests.

This fixture is intentionally tiny:
- `0Glu8uEj` provides the raw MOMENTS clip metadata and transcript JSON
- `0jJj5Mme` provides an extracted 10-frame frame sequence

The smoke test uses these together to validate:
- path resolution across local checkout and repo-relative paths
- prompt construction
- 10-frame loading into a single `VLPrompt`
