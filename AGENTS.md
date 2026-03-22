# AGENTS.md for Rust numerical assignments

This repo contains a sequence of related Rust numerical-methods assignments.

Before editing:
1. Inspect the closest completed assignments and identify the local patterns for:
   - crate / module layout
   - naming
   - helper reuse
   - float formatting
   - plot / data-export workflow
   - comment style
   - tests
2. Match those patterns in your implementation.

Hard requirements:
- Solve the full assignment. Do not omit graduate / optional parts when the user explicitly requires them.
- Prefer extending the repo's existing style over introducing a new architecture.
- Reuse existing utilities before creating new helpers.
- Regenerate outputs from code; do not hand-edit output files.
- Before finishing, run the repo's normal verification flow plus cargo fmt, cargo clippy, and cargo test.
- In the final response, state:
  - which existing files were used as style references,
  - which files changed,
  - where generated outputs were written,
  - and what verification commands were run.
