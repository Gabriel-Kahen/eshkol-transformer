# CLI3 installed corpus validate blocker

Status: source-only integration audit on `29d0db2`. The standalone D1 command
in `cli/eshkol-transformer.esk` is not part of the installed production CLI3
binary. No compiled or hosted gate has been run for this union.

## Source boundary

- `scripts/build-cli3.sh` compiles `src/eshkol_transformer/cli.esk` against
  `native/cli3_generate_root.esk`. That entry calls only the CLI3 dispatch and
  fallback exports. `native/cli3_generate_public_exports.txt` allows exactly
  those two exports; `native/cli3_defined_symbols.txt` lists them and six E1B
  error exports. The production package does not export the D1 public facade.
- `lib/transformer/data.esk` defines the public `token-corpus-validate` and six
  summary accessors using `et_e1b_public_d1_*` bridge symbols. The complete
  module also contains D2 wrappers. The standalone gate extracts the existing
  D1-only marked section to avoid linking D2; production CLI3 has no such
  public D1-only entry module.
- Production `native/cli3_extension.esk` implements `corpus inspect` through
  `et-d1-private-token-corpus-validate`, with option-form input, JSON output,
  smaller default limits, and CLI3 exit/diagnostic conventions. The standalone
  `corpus validate DIR` contract requires a positional path, a plain one-line
  summary, 64 MiB/64 MiB/1 billion defaults, usage exit 64, and structured D1
  failure exit 2. Aliasing `inspect` would change the accepted contract and
  bypass the standalone command's public-D1-only requirement.
- Required `scripts/test-cli3-ci.sh` invokes production generation and the
  cross-command chain. Neither invokes installed `corpus validate`.

## Next source contract and gates

A reviewed package/link contract must admit the existing D1 public validator
and six summary accessors into the installed executable without exposing
unreviewed D2 APIs or changing the existing CLI3 commands. Then add a narrow
installed dispatch preserving the standalone argument, output, error, and exit
contract, plus a required command test on the installed binary. Update exact
export/defined/undefined/source-closure manifests as that contract requires.
Run the focused source checks, a fresh composed-head pinned-toolchain D1/CLI3
compiled gate, and exact-head hosted CI before claiming production admission.

This audit does not establish a missing D1 semantic API. It identifies the
current CLI3 package boundary and command-contract differences that prevent a
source-only dispatch alias from being a valid implementation.
