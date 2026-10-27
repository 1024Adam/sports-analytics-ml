# Security Policy

## Supported Versions

Security updates are applied to the latest `main` branch of `sports-analytics-ml`.

| Version | Supported          |
| ------- | ------------------ |
| `0.1.x` (`main`) | :white_check_mark: |

---

## Reporting a Vulnerability

If you discover a security vulnerability within this repository, please **do not** open a public GitHub issue.

Instead, please report it privately using [**GitHub Private Vulnerability Reporting**](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) under the repository's **Security -> Advisories** tab.

Please include:
1. A description of the issue and its potential impact.
2. Steps or a minimal reproduction snippet to demonstrate the issue.
3. Any suggested remediation or mitigation, if applicable.

---

## Data, Credentials & Artifact Hygiene

Contributors and users working with upstream sports feeds and model artifacts should follow these security practices:

1. **No Secrets or API Keys in Source Control**
   - Never commit API keys, session cookies, proxy credentials, or `.env` files to the repository.
2. **External Feed Deserialization, Archive Extraction & Compression**
   - Treat all responses from upstream endpoints (Official NHL Web API, MoneyPuck CSV/ZIP archives, Official FPL API, Understat, Football-Data.co.uk) as untrusted input. Validate payloads through typed `pydantic` contracts or explicit Polars schema enforcement before persisting to Silver Parquet tables.
   - Archive extraction and decompression (such as MoneyPuck `.zip` bundles and `.csv.gz` / `.json.gz` raw cache streams in `core/cache.py`) must guard against path traversal (Zip Slip), unexpected archive members, and decompression bombs. Raw cache compaction (`sports-ml compact-cache`) writes atomically via temporary files and verifies compressed payload readability before unlinking legacy files.
3. **Model Artifacts & Local Caches**
   - Local data caches (`data/`) and trained model bundles (`artifacts/`) are git-ignored by default. Only load serialized model bundles (`joblib` / `pickle` / LightGBM models) that were generated from trusted local runs or verified SHA-256 schema fingerprints (`metadata.json`).
