# Binius64 recursion: a committed-table discharge of the deferred monster evaluation

`binius-recursion-discharge` makes the Binius64 IOP verifier **succinct under
recursion**. When many leaf proofs of a fixed constraint-system shape are aggregated,
each leaf's verifier defers one O(circuit-size) "monster" evaluation to a public wire via
`compute_public_value`; the final verifier would otherwise recompute that evaluation
natively, once per leaf. This crate commits the constraint-term table once, as a
per-shape verification key, and discharges all K deferred monster claims with **one**
batched argument whose dominant verifier cost is a fixed FRI/opening endgame — independent
of both the constraint-system size N and the batch size K. It introduces no new PCS, IOP,
or field: the argument is assembled entirely from primitives already in this repository
(`sumcheck::batch_verify`, `sumcheck::verify` over a bivariate product,
`fracaddcheck::verify`, `verify_mlecheck_basefold`, `fri::encode_interleaved` +
`merkle_tree::commit_field_buffer`) plus the
leaf verifier's `compute_public_value` deferral hook.

## Protocol summary

**Why the verifier is O(N).** The Binius64 IOP verifier closes its shift reduction in
`check_eval` (`crates/verifier/src/protocols/shift`) by asserting
`witness_eval · monster_eval == eval`, where `monster_eval(c)` is the batched multilinear
evaluation of the constraint matrices at a public-coin claim point `c`. This evaluation
costs one GF(2^128) multiplication per constraint term and is essentially the entire
per-proof arithmetic cost of the verifier: on 26 real leaf proofs we measured a **median
of 51.7M GF(2^128) multiplications per verify (range 12.5M–101M), of which ~99.93% is the
monster**. Everything else in `IOPVerifier::verify` — the intmul GKR, the bit-and and
shift sumchecks, ring-switch, and BaseFold/FRI — is polylog in N.

**The deferral hook, and the gap it leaves.** `compute_public_value`
(`crates/ip/src/channel.rs`) hoists `monster_eval` out of the recursion circuit: in the
`IronSpartan` ZK-wrap the closure is dropped and the result becomes a single inout wire,
so the O(N) work disappears from the *circuit* and is instead recomputed natively by
whoever verifies the outer proof. This makes the circuit succinct but not the verifier —
for K aggregated leaves the final verifier still pays K native monster passes, i.e.
K·O(N). Because the deferred wire is a free prover input to the in-circuit assertion, it
must also be *pinned* to its true value somewhere, or the leaf's `check_eval` becomes
vacuous (see SPEC §0/§5.1). Closing that gap soundly is what this crate does.

**The discharge.** Fix one constraint-system shape S, whose term table is the flattened
list of N tuples `(x_t, y_t, s_t, op_t, m_t)`, one per `ShiftedValueIndex` occurrence
(the value-vector index `y_t` addresses the two-segment public/hidden value vector; see
the SPEC's segmented-Y note). For an all-BitAnd shape the deferred value re-associates
into a **product of three multilinears** over the N terms,

```
monster_eval(c) = Σ_t  E_x(c)[t] · E_y(c)[t] · E_g(c)[t]
```

— a degree-3 sumcheck claim. Given K same-shape claims `{(c_ℓ, v_ℓ)}`, the discharge
reduces all K with one transcript:

- **Phase A** — one batched degree-3 sumcheck over the N terms (`sumcheck::batch_verify`)
  reduces the K claims to a shared point ρ and, via the marginalization and eight-point
  identities (SPEC §1), to O(K) evaluation claims on a *single* histogram oracle `M_D`
  (the ρ-weighted term histograms of the table's `x`/`y`/meta columns).
- **Phase B** — one bivariate-product sumcheck (`sumcheck::verify`) collapses those O(K)
  `M_D` claims to a single point σ. This is the only K·O(|point|) work the verifier does.
- **Phase C** (STEP 2) — one *weighted* `fracaddcheck::verify` certifies that `M_D` really
  is the ρ-histogram of the committed table columns `M_VK`, by a coefficient-matching
  partial-fraction identity with coset-disjoint block tags (SPEC §2, §1.3).
- **Final check** — **STEP 1** rebuilds `M_D` natively from the constraint system (whose
  digest it verified) in one O(N) table pass and checks `M̃_D(σ) == m`; this collapses the
  final verifier's `K × (monster + rebuild)` into a single O(N) pass. **STEP 2** replaces
  that native pass with **one merged, non-ZK batched BaseFold opening** of `[M_VK, M_D]`
  (a batched degree-2 reduction sumcheck over both oracles, then one combined
  MLE-check + FRI, with both Merkle roots pinned — `M_VK`'s to the audited `vk_digest`,
  `M_D`'s to the pre-`τ` `digest_D`), so the verifier's dominant cost becomes a fixed
  FRI/opening endgame, **independent of N and K**, above a sub-dominant K·polylog
  residual.

The committed table `M_VK = [X | Y | U | 0]` is a per-shape verification key, committed
once (`fri::encode_interleaved` + `merkle_tree::commit_field_buffer`); its digest
**is** the vkey. The discharge uses no ZK
masking anywhere. The built instantiation targets an all-BitAnd constraint shape
(`T_mul = ∅`, enforced by an admission check) and one flat aggregation level (K same-shape
leaves → one proof); the general two-lane extension and multi-level trees are described but
not built. See [`SPEC-monster-discharge.md`](./SPEC-monster-discharge.md) for the
full algebra, the soundness chain (Fiat-Shamir binding of the committed table and the
char-2 partial-fraction argument) and the error budget. Note: the SPEC predates upstream's
segmented value vector and its own STEP-2 endgame port, so where it describes a flat Y
column or two separate BaseFold openings, this crate implements the segmented Y histogram
and the single merged opening described above; the code is the source of truth.

## Build and test

The crate builds and tests as an ordinary workspace member; STEP-1 (native final check)
and STEP-2 (committed-table PCS endgame) both run under the default feature set.

```sh
# Full suite (STEP 1 + STEP 2 + adversarial negatives):
cargo test --release -p binius-recursion-discharge

# Real-capture cross-validation and negatives, with the per-test log lines:
cargo test --release -p binius-recursion-discharge -- --nocapture

# Synthetic scaling demo (reproduces the flat-in-N shape of the STEP-2 endgame):
cargo run --release -p binius-recursion-discharge --features test-utils --bin scaling_demo
```

The `test-utils` feature (enabled automatically for this crate's tests) gates the
synthetic constraint-system generators, standalone claim synthesis, and the adversarial
prover entry points; it is not part of the library API. The suite includes a bit-for-bit
cross-validation of the natively derived segmented term-sum against the real
`Verifier::verify` on genuine leaf proofs, plus adversarial negatives: table-swap,
tampered value, tampered Phase-A sum, a consistent table lie caught only by the committed
verification key, a corner-value forgery caught only by the merged opening, and a lie in
the unused `M_D` block that is invisible to both sumcheck phases and is caught by the
STEP-1 native final check.

## Measured results

Absolute figures below are from **real Binius64 leaf proofs** on an M2 laptop, for a fixed
shape with `N = 24,470,148` terms (`N_pad = 2^25`), K = 3 distinct same-shape leaves;
independently reproduced. The in-tree tests reproduce the *shape* of these results (the
flat-in-N STEP-2 endgame, the growing STEP-1 pass) on synthetic tables at CI-friendly
sizes — see `scaling_demo`.

| metric | result |
|---|---|
| native leaf verify | median **51.7M** GF(2^128) mults (26 proofs, 12.5M–101M); **~99.93% is the monster** |
| **STEP-2 discharge verify** | **~4 ms** at K=3; dominant FRI/opening endgame **flat in N** (N=15 → ~1 ms, N=24.5M → ~4 ms; FRI log-depth only) and **K-independent** by construction, above a sub-dominant K·polylog residual |
| STEP-1 discharge verify | one O(N) native table pass, done **once** per batch (K-independent) in place of K native monster passes |
| **integrated wrap verify** (K=3, one FS stream) | **~18 ms** vs **~361 ms** native re-verification (≈20×; gap grows with N·K). Measured on a separate integrated-wrap crate not included in this PR (it drives the discharge from the outer aggregation); this figure is not reproducible from this crate alone. |

The succinctness claim is on the *final verifier*: the prover remains O(K·N) in streaming
multiplications, but all committed data, proof size, and final-verifier cost are
K-independent, and the STEP-2 verifier is additionally N-independent up to the FRI
log-depth.

## Reference

- [`SPEC-monster-discharge.md`](./SPEC-monster-discharge.md) — the full protocol
  specification: algebraic decomposition, the four-phase reduction, the soundness chain and
  error budget, and the trust roots. It was written against an earlier commit and predates
  two upstream changes the code has since absorbed — the **segmented value vector**
  (the Y column is a two-segment public/hidden histogram, not a flat `eq(y, r_y)`; the
  claim point carries a trailing `r_segment`) and the **merged STEP-2 opening** (one
  batched `[M_VK, M_D]` opening, not two). Where the SPEC and the code differ on these two
  points, the code is authoritative; file paths and function names remain the reliable
  cross-reference for everything else.
