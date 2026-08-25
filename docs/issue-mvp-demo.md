# PR Issue: Add Live Demo + Visual Proof + Try It Section (MVP)

## Summary
Add a real demo experience to the repo landing page by surfacing a live demo URL, a screenshot/GIF of the UI?prediction flow, and a minimal ?Try it? block with sample inputs/expected output in the README.

## Why
This is the top MVP differentiator: it upgrades the repo from ?code? to ?product-like? and makes it recruiter-ready.

## Acceptance Criteria
- README includes a **Live demo** link near the top (above the fold).
- README includes **1 screenshot or a 10?15s GIF** showing input ? prediction result.
- README includes a **?Try it?** section with:
  - example input values
  - expected output format and sample value (currency/units)
- If FastAPI is the demo, the README links to `/docs` for the API schema.

## Suggested Implementation
- Add a ?Demo? block near the top of `README.md` with the URL and image/GIF.
- Add a ?Try it? block with a copy?paste example (either JSON or form inputs).
- Store image/GIF in `docs/` (or `assets/`) and reference it in README.

## Notes
- If no live demo is available yet, add a placeholder and a TODO date, but prefer a minimal hosted endpoint.

## Checklist
- [ ] Live demo URL added
- [ ] Screenshot/GIF added
- [ ] Try it section added
