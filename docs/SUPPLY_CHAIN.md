# Software supply chain

## Source SBOM

The `sbom` job in `.github/workflows/ci.yml` uses Anchore's Syft-backed GitHub Action to generate an SPDX JSON inventory from the checked-out repository. CI retains the file as the `medai-source.spdx.json` workflow artifact. The separate `Prepare source release` workflow runs on `v*` tag pushes (or a manual rerun against that exact tag), packages the tagged commit as `medai-source.tar.gz`, generates its SBOM, and creates or updates a GitHub draft release with the archive, SBOM, and `SHA256SUMS`. Review the draft and its attestations, then publish it. Assets are attached before publication so repositories with immutable releases can use the workflow.

## Release attestations

The release job creates a keyless Sigstore build-provenance attestation for `medai-source.tar.gz` and an SPDX SBOM attestation that binds the same archive to `medai-source.spdx.json`. Attestations are stored by GitHub and are associated with the repository; they are not signing-key files attached to the release. Verify the downloaded files and provenance with:

```sh
sha256sum --check SHA256SUMS
gh attestation verify medai-source.tar.gz --repo OWNER/REPOSITORY
```

Use a current GitHub CLI with network access. The verification command checks the attestation signature and repository identity. Releases should be created from protected tags, and reviewers should confirm the attested source commit matches the intended release. GitHub documents artifact attestation availability for public repositories and for private/internal repositories on supported Enterprise Cloud plans; the release workflow will fail if repository settings or plan do not permit attestation writes.

The SBOM inventories dependency data present in the source tree; it is not proof that every deployed container image was built from that revision. The web package currently has no committed npm lockfile, so transitive JavaScript resolution is not yet reproducible or fully represented. Python dependencies are declared in `science/pyproject.toml`; Rust dependencies are resolved by the committed `Cargo.lock`.

## Rust dependency license gate

The `rust-license-policy` job runs cargo-deny against the committed `Cargo.lock` and `deny.toml`. Rust dependencies must resolve to an explicitly allowed license. The `r-efi` exception permits its LGPL option only for that crate; this does not allow LGPL dependencies generally. Review license metadata and source license files when updating dependencies. For legacy license metadata that is not an SPDX expression, cargo-deny can infer the license from packaged license text when its configured confidence threshold is met.

The Python CI environment also runs pinned `pip-licenses` metadata collection after installing the test, MCP, bioinformatics, and chemistry extras. A local policy checker requires every semicolon-separated license label for every installed package to match the allowlist; unknown, missing, compound, or otherwise unrecognized labels fail closed and require review. This checks packages installed by that job, including transitives. It does not yet resolve the optional `ml` and `rna` extras, and Python dependency versions are not locked; the result is a license gate for the CI environment, not a complete release SBOM or legal review. Keep the allowlist deliberately explicit and investigate any newly reported license before adding it.

## Remaining release controls

SBOM generation is inventory, not a complete software clearance. Frontend dependency license enforcement is also pending because the web package has no committed lockfile. The release attestation covers the source archive, not deployment images; image-specific SBOM publication and provenance remain open. Do not use the source SBOM alone as a compliance attestation.
