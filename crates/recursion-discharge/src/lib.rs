// Copyright 2026 The Binius Developers

//! Batched discharge of deferred Binius64 monster claims.
//!
//! When many Binius64 proofs of one fixed constraint-system shape are verified under
//! recursion, each leaf verifier defers its single O(circuit-size) "monster"
//! multilinear evaluation (`check_eval`'s `compute_public_value` in
//! `binius_verifier::protocols::shift`) to a public value. This crate certifies K such
//! deferred claims `v_l = monster(c_l)` with ONE batched argument instead of K native
//! monster evaluations:
//!
//! * Phase 0 — statement absorption + structural preconditions (the spec's P0.1–P0.4);
//! * Phase A — one batched degree-3 sumcheck over the term domain (K claims → one shared point
//!   rho);
//! * Phase C (STEP 2) — one weighted fracaddcheck certifying the committed `M_D` equals the
//!   rho-weighted histograms of the committed verification-key columns (union-domain partial
//!   fractions with coset-disjoint tags);
//! * Phase B — one bivariate sumcheck over `[W_eq, M_D]` (10K(+1) evaluation claims → one point
//!   sigma with claimed eval m);
//! * Final check — STEP 1: one native rebuild of `M_D` from the constraint system, asserting
//!   `M~_D(sigma) == m`; STEP 2: one merged non-ZK batched BaseFold opening of `[M_VK, M_D]` — the
//!   STEP-2 verifier never touches the constraint system (it takes only the verification-key
//!   metadata, the statement, and the transcript).
//!
//! # When to use
//!
//! Reach for this crate when a recursive aggregator verifies many same-shape Binius64
//! proofs and the per-leaf monster evaluations dominate the final verifier's cost.
//! STEP 1 removes the K-fold multiplicity (one native table pass per batch); STEP 2
//! additionally makes the final check independent of both the constraint-system size
//! and the batch size, up to FRI log-depth.
//!
//! # Key types
//!
//! * [`discharge::DischargeStatement`] — the batch statement: constraint-system digest, table
//!   metadata, and the K `(claim point, claimed value)` pairs.
//! * [`discharge::discharge_prove`] / [`discharge::discharge_verify`] — STEP-1 prover and verifier
//!   (the verifier holds the constraint system for the native final check).
//! * [`vk::vkgen`] / [`vk::DischargeVkm`] — deterministic per-shape verification-key generation and
//!   its metadata blob.
//! * [`step2::discharge_prove_step2`] / [`step2::discharge_verify_step2`] — STEP-2 prover and
//!   verifier (committed tables; constraint-system-free verifier).
//! * [`recorder::verify_and_capture`] — runs the plain leaf verification with a recording channel
//!   and captures the deferred monster claim.
//! * [`error::VerifyError`] — one named variant per verifier-side check.
//!
//! # Related crates
//!
//! Builds entirely from primitives in this workspace: `binius-ip`/`binius-ip-prover`
//! (sumcheck, fracaddcheck), `binius-iop`/`binius-iop-prover` (BaseFold/FRI, Merkle
//! channels), and `binius-verifier` (the leaf verifier whose monster claim is
//! discharged). See this crate's `README.md` for the protocol summary and measured
//! results, and `SPEC-monster-discharge.md` for the full algebra and soundness
//! analysis.

pub mod cubic;
pub mod discharge;
pub mod error;
pub mod fracadd;
#[cfg(feature = "test-utils")]
pub mod leaf;
pub mod merged;
pub mod packed;
pub mod recorder;
pub mod step2;
#[cfg(feature = "test-utils")]
pub mod synth;
pub mod table;
pub mod vk;

pub use binius_verifier::config::B128;
