# Privacy and Safety

pgoggles is designed for local-first experimentation with clear consent
boundaries. The default SDK and examples must not include scraper execution paths
or biometric identification workflows.

## Defaults

- No identity recognition runs in the default install.
- Example apps use local test data only.
- Recording, cloud upload, biometric matching, and person identification require
  explicit configuration.
- Optional research packages live under `research/` and are documented as such.

## Identity Research

The legacy face-vector work is preserved under `research/identity` for consented
local datasets and retrieval experiments. It is excluded from the default setup
unless `PGOGGLES_INSTALL_IDENTITY_RESEARCH=1` is set.

Any identity workflow should require `PGOGGLES_IDENTITY_RESEARCH_ENABLED=1` or an
equivalent explicit opt-in before it can run.

## Data Handling

- Prefer local processing and local storage.
- Document retention and deletion behavior for every app that records data.
- Make recording state visible in user-facing apps.
- Avoid collecting bystander data where possible.
