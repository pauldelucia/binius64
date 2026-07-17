// Copyright 2026 The Binius Developers

//! STEP-2 discharge: committed M_VK + per-batch committed M_D + Phase C
//! (weighted fracaddcheck over the (n_d + 2)-var union domain) + PCS final check
//! (ONE merged non-ZK batched BaseFold opening of [M_VK, M_D] — see `crate::merged`).
//! The verifier NEVER touches the ConstraintSystem — it takes (VKM, statement,
//! transcript) only (P0.2 checks the statement's cs_digest against the VKM's).
//!
//! Transcript layout (one FS transcript; spec section 3 challenge order):
//!   observe(VKM) | observe(statement) | Phase A [mu; rounds; 3K finish evals] |
//!   8K d values | digest_D | tau | d_root | fracaddcheck layers (num/den to point pi) |
//!   [v00 v10 v01 m_pi] | phi | Phase B rounds | m | rho_c |
//!   merged [M_VK, M_D] opening: M_VK commitment (root pinned to the audited
//!   vk_digest) | batched degree-2 reduction sumcheck (M_VK at [pi_lo, rho_c] with the
//!   corner-combined claim; M_D at sigma with claim m, top-padded) | two alphas |
//!   one outer challenge | combined MLE-check + FRI over BOTH pinned codewords.
//!
//! Union-domain layout (spec 1.1, selector-high): low n_l := n_d coords = t (blocks
//! 00/10/01 = X/Y/U rows) or M_D's own (a, c) index (block 11); blk at the TOP two
//! coords. num = [eq(t,rho_ext) | eq | eq | M_D]; den = [tau+X | tau+Y | tau+U |
//! tau+emb], emb(w) = B128(w) under the aligned tag basis (vk.rs). The total
//! fractional sum is identically 0 for the honest histograms (char-2 pole
//! cancellation); coset-disjoint tags make partial fractions force M_D = the
//! rho-weighted histograms, pole family by pole family.

use anyhow::ensure;
use binius_core::constraint_system::ConstraintSystem;
use binius_field::Field;
use binius_hash::StdHashSuite;
use binius_iop::{
	merkle_channel::{
		MerkleIPVerifierChannel, TranscriptMerkleCommitment, VerifierMerkleTranscriptChannel,
	},
	merkle_tree::Commitment,
};
use binius_iop_prover::{
	fri::encode_interleaved,
	merkle_channel::{MerkleIPProverChannel, ProverMerkleTranscriptChannel},
};
use binius_ip::{
	channel::IPVerifierChannel,
	fracaddcheck::{self, FracAddEvalClaim},
	prodcheck::MultilinearEvalClaim,
};
use binius_ip_prover::{
	channel::IPProverChannel,
	sumcheck::{
		batch::batch_prove_and_write_evals, bivariate_product_evaluator::bivariate_product_prover,
		prove_single,
	},
};
use binius_math::{
	FieldBuffer,
	multilinear::{eq::eq_ind_partial_eval, evaluate::evaluate},
	univariate::evaluate_univariate,
};
use binius_transcript::{ProverTranscript, VerifierTranscript, fiat_shamir::Challenger};
use binius_verifier::{config::B128, protocols::shift::SHIFT_VARIANT_COUNT};

use crate::{
	cubic::CubicProductSumcheckProver,
	discharge::{DischargeStatement, axpy_point_tensor, claim_contexts_from_dims},
	error::{VerifyError, verify_ensure},
	fracadd::FastFracAddProver,
	merged::OpenClaim,
	packed::{
		PB, assemble_m_d_packed, axpy_dense_par, build_m_vk_packed, build_phase_c_leaf_halves,
		gather_column, vk_corner_values_packed,
	},
	table::{TermTable, build_histograms, eq_point, evaluate_d_g, extract_table, m_d_points},
	vk::{Digest, DischargeVkm, beta, build_ntt, build_pcs, encode_m_vk},
};

mod tamper {
	/// Adversarial knobs for the STEP-2 prover (`None` = honest). The tampered prover
	/// follows the honest machinery on tampered data — the strongest adaptive cheat the
	/// adversarial suite exercises. Public only with the `test-utils` feature.
	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum Step2Tamper {
		None,
		/// +1 at one slot of M_D's (1,1) selector block BEFORE the M_D commit. Invisible
		/// to Phases A and B (W_eq vanishes there and m/m_pi are consistently tampered);
		/// the weighted fracaddcheck (Phase C) is the only check that can reject it.
		MdBlock3,
		/// Commit the HONEST M_D (so digest_D is FS-consistent on both sides and every
		/// transcript phase passes), but hand the merged opener an M_D codeword that no
		/// longer matches that committed digest. The upstream channel API gives no way to
		/// write a flipped root while opening the real tree (the commitment handle is
		/// only obtainable from `send_merkle_commitment`, which writes the true root), so
		/// the digest-vs-polynomial mismatch is realized on the codeword the FRI opens:
		/// every query leaf then fails to authenticate against the honest committed root
		/// — the M_D opening's Merkle verification rejects.
		DigestD,
	}
}
#[cfg(feature = "test-utils")]
pub use tamper::Step2Tamper;
#[cfg(not(feature = "test-utils"))]
pub(crate) use tamper::Step2Tamper;

/// Shared structural preconditions (P0.1/P0.2 metadata coherence, statement side).
fn check_statement_against_vkm(
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
) -> anyhow::Result<()> {
	ensure!(
		stmt.cs_digest == vkm.cs_digest,
		"P0.2 cs_digest mismatch: statement {} vs VKM {}",
		hex(&stmt.cs_digest),
		hex(&vkm.cs_digest)
	);
	ensure!(
		stmt.n_terms == vkm.dims.n_terms
			&& stmt.n_pad == vkm.dims.n_pad
			&& stmt.parity == vkm.dims.parity,
		"P0.1 table metadata mismatch: statement (N={}, N_pad={}, parity={}) vs VKM (N={}, N_pad={}, parity={})",
		stmt.n_terms,
		stmt.n_pad,
		stmt.parity,
		vkm.dims.n_terms,
		vkm.dims.n_pad,
		vkm.dims.parity,
	);
	ensure!(!stmt.claims.is_empty(), "empty claim batch");
	Ok(())
}

fn hex(bytes: &[u8]) -> String {
	bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The Phase-B claim list shared by prover and verifier: per claim l the 10 M_D points
/// with evals [a_l, b_l, d_{l,0..8}], plus the Phase-C point (pi_lo, m_pi) LAST.
fn phase_b_claims(
	dims: &crate::table::ShapeDims,
	ctxs: &[crate::discharge::ClaimCtx],
	finish_evals: &[B128],
	d_vals: &[B128],
	pi_lo: &[B128],
	m_pi: B128,
) -> (Vec<Vec<B128>>, Vec<B128>) {
	let k = ctxs.len();
	let mut points = Vec::with_capacity(10 * k + 1);
	let mut evals = Vec::with_capacity(10 * k + 1);
	for (l, ctx) in ctxs.iter().enumerate() {
		points.extend(m_d_points(dims, &ctx.tr));
		evals.push(finish_evals[3 * l]);
		evals.push(finish_evals[3 * l + 1]);
		evals.extend_from_slice(&d_vals[l * SHIFT_VARIANT_COUNT..(l + 1) * SHIFT_VARIANT_COUNT]);
	}
	let mut pi_point = pi_lo.to_vec();
	// The pi point addresses M_D's full n_d-var domain directly (address + its own
	// selector pair); it is already n_d coordinates.
	debug_assert_eq!(pi_point.len(), dims.n_d);
	points.push(std::mem::take(&mut pi_point));
	evals.push(m_pi);
	(points, evals)
}

/// STEP-2 prover. The prover holds the CS (it is the table's source) and the VKM.
pub fn discharge_prove_step2<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
) -> anyhow::Result<()> {
	discharge_prove_step2_impl(cs, vkm, stmt, transcript, Step2Tamper::None)
}

/// Test-only entry with adversarial knobs. Follows the honest machinery on tampered
/// data (see [`Step2Tamper`]).
#[cfg(feature = "test-utils")]
pub fn discharge_prove_step2_tampered<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<()> {
	discharge_prove_step2_impl(cs, vkm, stmt, transcript, tamper)
}

fn discharge_prove_step2_impl<C: Challenger>(
	cs: &ConstraintSystem,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<()> {
	let table = extract_table(cs)?;
	ensure!(
		table.cs_digest == vkm.cs_digest && table.dims == vkm.dims,
		"prover CS does not regenerate the VKM's shape (T1 hygiene)"
	);
	discharge_prove_step2_on_table_impl(&table, &table, vkm, stmt, transcript, tamper)
}

/// STEP-2 prover over a CALLER-SUPPLIED term table (no CS extraction, no T1 hygiene
/// check against the table contents — the table is trusted as given). Intended for
/// adversarial test harnesses that prove over a deliberately tampered table (the
/// "consistent lie" that Phases A/B cannot see and only the committed-table binding
/// rejects). `table` drives Phases A/B/C data; `vk_table` drives the M_VK re-commit
/// and opening (a realistic adversary keeps it honest so the re-committed digest
/// matches the pinned vk_digest, and is then caught by the opening's false claim).
/// Honest callers use [`discharge_prove_step2`], which passes the same table for both.
#[cfg(feature = "test-utils")]
pub fn discharge_prove_step2_on_table<C: Challenger>(
	table: &TermTable,
	vk_table: &TermTable,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<()> {
	discharge_prove_step2_on_table_impl(table, vk_table, vkm, stmt, transcript, tamper)
}

fn discharge_prove_step2_on_table_impl<C: Challenger>(
	table: &TermTable,
	vk_table: &TermTable,
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut ProverTranscript<C>,
	tamper: Step2Tamper,
) -> anyhow::Result<()> {
	check_statement_against_vkm(vkm, stmt)?;
	let mut ctxs = claim_contexts_from_dims(&table.dims, stmt.parity, stmt, false)?;
	let dims = &table.dims;
	let n_l = dims.n_d;
	let k = ctxs.len();

	let pcs = build_pcs(vkm)?;
	let ntt = build_ntt(pcs.log_domain);

	// Phase 0: VKM + statement absorption (P0.1) — before mu is sampled.
	IPProverChannel::<B128>::observe_many(transcript, &vkm.to_elems());
	IPProverChannel::<B128>::observe_many(transcript, &stmt.to_elems());

	// ---- Phase A: K cubic sumchecks via the upstream batch driver. ----
	let phase_a_guard = tracing::info_span!("[phase] Discharge A", k).entered();
	let mut provers = Vec::with_capacity(k);
	for ctx in &ctxs {
		provers.push(CubicProductSumcheckProver::new(
			[
				gather_column::<PB, _>(&table.terms, &ctx.tr.x_tensor, dims.n_t, |t| t.x as usize),
				gather_column::<PB, _>(&table.terms, &ctx.tr.y_tensor, dims.n_t, |t| t.y as usize),
				gather_column::<PB, _>(&table.terms, &ctx.tr.g_tab, dims.n_t, |t| t.u as usize),
			],
			ctx.sum,
		));
	}
	let output = batch_prove_and_write_evals(provers, transcript);
	let rho = output.challenges.clone(); // driver returns low-first
	ensure!(rho.len() == dims.n_t, "phase A round count");
	drop(phase_a_guard);

	// ---- Histograms at rho + the 8K d values. ----
	let hist_guard = tracing::debug_span!("Build histograms").entered();
	let hist = build_histograms(table, &rho);
	let mut d_vals = Vec::with_capacity(k * SHIFT_VARIANT_COUNT);
	for ctx in &ctxs {
		let points = m_d_points(dims, &ctx.tr);
		for point in &points[2..] {
			d_vals.push(evaluate_d_g(&hist.d_g, &point[..crate::table::N_U]));
		}
	}
	IPProverChannel::<B128>::send_many(transcript, &d_vals);
	// Free the per-claim eq tensors before the memory-heavy Phase C (only O(arity)
	// transparents — parsed points, h_ops, m_coords — are needed from here on).
	for ctx in &mut ctxs {
		ctx.tr.x_tensor = Vec::new();
		ctx.tr.y_tensor = Vec::new();
		ctx.tr.g_tab = Vec::new();
	}
	let ctxs = ctxs; // immutable from here
	drop(hist_guard);

	// ---- Commit M_D (oracle 1 of the batched params); digest observed BEFORE tau. ----
	let commit_d_guard = tracing::info_span!("Commit M_D").entered();
	let mut m_d = assemble_m_d_packed::<PB>(dims, &hist);
	if tamper == Step2Tamper::MdBlock3 {
		let idx = (3usize << dims.n_a) + (12345 % (1usize << dims.n_a));
		let cur = m_d.get(idx);
		m_d.set(idx, cur + B128::ONE);
	}
	// Encode M_D as oracle 1 of the batched params, then commit it over a Merkle channel:
	// `send_merkle_commitment` writes digest_D as an OBSERVED message (before tau, P0.1)
	// and returns the opening handle held for the merged FRI.
	let m_d_codeword = encode_interleaved(&pcs.params, 1, &ntt, m_d.to_ref());
	let d_commitment = {
		let mut mchan =
			ProverMerkleTranscriptChannel::<_, C, B128, StdHashSuite>::new(&mut *transcript);
		mchan.send_merkle_commitment(m_d_codeword.to_ref(), pcs.leaf_size(1))
	};
	// DigestD adversary: digest_D above is HONEST (so the FS stream is consistent and every
	// transcript phase passes), but the codeword handed to the merged opener is tampered on
	// EVERY leaf, so each FRI query fails to authenticate against the honest committed root
	// (Merkle rejection). The upstream channel API gives no way to write a flipped root while
	// opening the real tree, so the digest↔polynomial mismatch is realized here instead.
	let m_d_codeword_fri = if tamper == Step2Tamper::DigestD {
		let mut c = m_d_codeword;
		for i in 0..(1usize << c.log_len()) {
			c.set(i, c.get(i) + B128::ONE);
		}
		c
	} else {
		m_d_codeword
	};
	drop(commit_d_guard);

	// ---- Phase C: weighted fracaddcheck over the (n_l + 2)-var union domain. ----
	let phase_c_guard = tracing::info_span!("[phase] Discharge C").entered();
	let tau: B128 = IPProverChannel::<B128>::sample(transcript);
	let leaf = {
		let _guard = tracing::debug_span!("Build Phase C leaf").entered();
		let eq_rho = eq_ind_partial_eval::<PB>(&rho);
		build_phase_c_leaf_halves(table, tau, &eq_rho, &m_d)?
		// eq_rho dropped here, before the layered tree doubles the footprint
	};
	let (frac_prover, (num_root, den_root)) = {
		let _guard = tracing::debug_span!("Build fraction tree").entered();
		FastFracAddProver::new(leaf)
	};
	if tamper != Step2Tamper::MdBlock3 {
		ensure!(num_root == B128::ZERO, "phase C tree root numerator nonzero (table/M_D mismatch)");
	}
	ensure!(den_root != B128::ZERO, "phase C tree root denominator zero");
	IPProverChannel::<B128>::send_one(transcript, den_root);
	let root_claim = (
		MultilinearEvalClaim {
			eval: B128::ZERO,
			point: Vec::new(),
		},
		MultilinearEvalClaim {
			eval: den_root,
			point: Vec::new(),
		},
	);
	let (num_leaf_claim, _den_leaf_claim) = {
		let _guard = tracing::debug_span!("Prove fraction layers").entered();
		frac_prover.prove(root_claim, transcript)
	};
	let pi = num_leaf_claim.point;
	ensure!(pi.len() == n_l + 2, "phase C leaf point arity");
	let pi_lo = &pi[..n_l];

	// Corner values + m_pi (recv'd by the verifier; validated at the PCS step /
	// Phase B respectively).
	let (v_corner, m_pi) = {
		let _guard = tracing::debug_span!("VK corner values").entered();
		let eq_pi = eq_ind_partial_eval::<PB>(pi_lo);
		(vk_corner_values_packed(table, &eq_pi), evaluate(&m_d, pi_lo))
	};
	IPProverChannel::<B128>::send_many(transcript, &[v_corner[0], v_corner[1], v_corner[2], m_pi]);
	drop(phase_c_guard);

	// ---- Phase B: one bivariate sumcheck over [W_eq, M_D], 10K + 1 claims. ----
	let phase_b_guard = tracing::info_span!("[phase] Discharge B").entered();
	let phi: B128 = IPProverChannel::<B128>::sample(transcript);
	let finish_evals: Vec<B128> = output
		.multilinear_evals
		.iter()
		.flat_map(|evals| {
			assert_eq!(evals.len(), 3, "cubic finish arity");
			evals.iter().copied()
		})
		.collect();
	let (points, evals) = phase_b_claims(dims, &ctxs, &finish_evals, &d_vals, pi_lo, m_pi);
	let combined = evaluate_univariate(&evals, phi);

	let mut w_eq = vec![B128::ZERO; 1 << dims.n_d];
	let mut phi_pow = B128::ONE;
	for p in points.iter().take(10 * k) {
		axpy_point_tensor(&mut w_eq, dims.n_a, p, phi_pow);
		phi_pow *= phi;
	}
	{
		// The Phase-C point is dense (no zero tail / 0-1 selectors): full tensor,
		// expanded and accumulated packed-parallel.
		let pi_tensor = eq_ind_partial_eval::<B128>(pi_lo);
		axpy_dense_par(&mut w_eq, pi_tensor.as_ref(), phi_pow);
	}
	let prover_b =
		bivariate_product_prover([FieldBuffer::<PB>::from_values(&w_eq), m_d.clone()], combined);
	drop(w_eq);
	let out_b = prove_single(prover_b, transcript);
	let mut sigma = out_b.challenges;
	sigma.reverse(); // low-first
	ensure!(sigma.len() == dims.n_d, "phase B round count");
	let m = out_b.multilinear_evals[1];
	IPProverChannel::<B128>::send_one(transcript, m);
	drop(phase_b_guard);

	// ---- Final check: ONE merged batched opening of [M_VK, M_D]. ----
	// rho_c is sampled AFTER m (the M_D claim) and the corner values are FS-bound, so
	// both point-evaluation claims are fixed before the opening begins.
	let rho_c: Vec<B128> = IPProverChannel::<B128>::sample_many(transcript, 2);
	let recommit_guard = tracing::info_span!("Re-commit M_VK").entered();
	// Rebuild M_VK from the CS-derived table (spec section 4) and encode it (oracle 0). The
	// channel commit below re-emits its Merkle root; for the HONEST prover this equals the
	// audited vk_digest, which the verifier asserts (the T1 / table-swap binding), so no
	// separate prover-side re-commit assert is needed.
	let m_vk = build_m_vk_packed::<PB>(vk_table);
	let vk_codeword = encode_m_vk(&pcs, &ntt, &m_vk);
	drop(recommit_guard);

	let open_guard = tracing::info_span!("Merged [M_VK, M_D] opening").entered();
	let claim_vk = eq_point(&[B128::ZERO, B128::ZERO], &rho_c) * v_corner[0]
		+ eq_point(&[B128::ONE, B128::ZERO], &rho_c) * v_corner[1]
		+ eq_point(&[B128::ZERO, B128::ONE], &rho_c) * v_corner[2];
	let mut vk_point = pi_lo.to_vec();
	vk_point.extend_from_slice(&rho_c);
	{
		// One Merkle channel drives the whole opening: commit M_VK (writes its root — bound
		// by the verifier to the audited vk_digest), then reduction + combine + the native
		// combined FRI over BOTH pinned codewords (see merged.rs / upstream
		// verify_mlecheck_basefold).
		let mut mchan =
			ProverMerkleTranscriptChannel::<_, C, B128, StdHashSuite>::new(&mut *transcript);
		let vk_commitment = mchan.send_merkle_commitment(vk_codeword.to_ref(), pcs.leaf_size(0));
		crate::merged::prove_merged_openings(
			&pcs.params,
			&ntt,
			&mut mchan,
			[
				OpenClaim {
					message: m_vk,
					point: &vk_point,
					eval: claim_vk,
				},
				OpenClaim {
					message: m_d,
					point: &sigma,
					eval: m,
				},
			],
			vec![
				(vk_codeword, vk_commitment),
				(m_d_codeword_fri, d_commitment),
			],
		)?;
	}
	drop(open_guard);

	Ok(())
}

/// STEP-2 verifier: (VKM, statement, transcript) — NO ConstraintSystem anywhere.
pub fn discharge_verify_step2<C: Challenger>(
	vkm: &DischargeVkm,
	stmt: &DischargeStatement,
	transcript: &mut VerifierTranscript<C>,
) -> Result<(), VerifyError> {
	let precondition = |e: anyhow::Error| VerifyError::Precondition(format!("{e:#}"));
	check_statement_against_vkm(vkm, stmt).map_err(precondition)?;
	let dims = &vkm.dims;
	let n_l = dims.n_d;
	// LIGHT contexts: the verifier's per-claim transparents are O(arity) mults — no
	// eq-tensor expansion, no O(2^n_x) term anywhere in this function.
	let ctxs = claim_contexts_from_dims(dims, stmt.parity, stmt, true).map_err(precondition)?;
	let k = ctxs.len();
	let pcs = build_pcs(vkm).map_err(precondition)?;

	// Phase 0 (P0.1): observe the full VKM blob, then the statement, before mu.
	IPVerifierChannel::<B128>::observe_many(transcript, &vkm.to_elems());
	IPVerifierChannel::<B128>::observe_many(transcript, &stmt.to_elems());

	// ---- Phase A. ----
	let sums: Vec<B128> = ctxs.iter().map(|c| c.sum).collect();
	let out_a = binius_ip::sumcheck::batch_verify::<B128, _>(dims.n_t, 3, &sums, transcript)
		.map_err(|e| VerifyError::Sumcheck {
			phase: "phase A batch_verify",
			source: e,
		})?;
	let mu = out_a.batch_coeff;
	let e_a = out_a.eval;
	let mut rho = out_a.challenges;
	rho.reverse(); // low-first

	let finish_evals: Vec<B128> =
		IPVerifierChannel::<B128>::recv_many(transcript, 3 * k).map_err(|e| {
			VerifyError::Channel {
				phase: "phase A finish evals",
				source: e,
			}
		})?;
	let prods: Vec<B128> = finish_evals
		.chunks_exact(3)
		.map(|abg| abg[0] * abg[1] * abg[2])
		.collect();
	verify_ensure!(evaluate_univariate(&prods, mu) == e_a, VerifyError::PhaseARecombination);

	let d_vals: Vec<B128> =
		IPVerifierChannel::<B128>::recv_many(transcript, SHIFT_VARIANT_COUNT * k).map_err(|e| {
			VerifyError::Channel {
				phase: "phase A d values",
				source: e,
			}
		})?;
	for (l, ctx) in ctxs.iter().enumerate() {
		let g_l = finish_evals[3 * l + 2];
		let gammas = ctx.tr.gammas();
		let mut rhs = B128::ZERO;
		for o in 0..SHIFT_VARIANT_COUNT {
			rhs += gammas[o] * d_vals[l * SHIFT_VARIANT_COUNT + o];
		}
		verify_ensure!(g_l == rhs, VerifyError::PhaseGRecombination { claim: l });
	}

	// ---- digest_D (M_D commitment; bound by FS before tau), then Phase C. ----
	let digest_d: Digest = transcript
		.message()
		.read()
		.map_err(|e| VerifyError::Transcript {
			phase: "digest_D read",
			source: e,
		})?;
	// Package M_D's PINNED commitment (oracle 1) for the merged opening; its depth and
	// leaf size come from the deterministic batch params (not from the prover).
	let d_commitment = TranscriptMerkleCommitment {
		commitment: Commitment {
			root: digest_d,
			depth: pcs.depth(1),
		},
		leaf_size: pcs.leaf_size(1),
	};
	let tau: B128 = IPVerifierChannel::<B128>::sample(transcript);
	verify_ensure!(tau.val() >= (1u128 << (dims.n_a + 2)).into(), VerifyError::TauInPoleRange);
	let d_root: B128 =
		IPVerifierChannel::<B128>::recv_one(transcript).map_err(|e| VerifyError::Channel {
			phase: "phase C d_root",
			source: e,
		})?;
	verify_ensure!(d_root != B128::ZERO, VerifyError::PhaseCRootDenominatorZero);
	let leaf_claim = fracaddcheck::verify::<B128, _>(
		n_l + 2,
		FracAddEvalClaim {
			num_eval: B128::ZERO,
			den_eval: d_root,
			point: Vec::new(),
		},
		transcript,
	)?;
	let pi = leaf_claim.point;
	verify_ensure!(pi.len() == n_l + 2, VerifyError::PhaseCLeafArity);
	let (pi_lo, pi_hi) = pi.split_at(n_l);

	let vvals: Vec<B128> =
		IPVerifierChannel::<B128>::recv_many(transcript, 4).map_err(|e| VerifyError::Channel {
			phase: "phase C corner values",
			source: e,
		})?;
	let (v00, v10, v01, m_pi) = (vvals[0], vvals[1], vvals[2], vvals[3]);

	// Transparent leaf-claim decompositions (spec section 2, Phase C).
	let zero = B128::ZERO;
	let one = B128::ONE;
	let eq_blk = [
		eq_point(&[zero, zero], pi_hi),
		eq_point(&[one, zero], pi_hi),
		eq_point(&[zero, one], pi_hi),
		eq_point(&[one, one], pi_hi),
	];
	let mut rho_ext = rho.clone();
	rho_ext.resize(n_l, zero);
	let num_expected =
		(eq_blk[0] + eq_blk[1] + eq_blk[2]) * eq_point(pi_lo, &rho_ext) + eq_blk[3] * m_pi;
	verify_ensure!(num_expected == leaf_claim.num_eval, VerifyError::PhaseCNumeratorTransparent);
	// col_11 = iota~(pi[..n_a]) + iota~'(pi[n_a..n_l]): the MLE of a linear map is the
	// map itself (plainly linear under the aligned basis). The X/Y/U columns carry their
	// kappa tags INSIDE the committed values, so col_00/10/01 are exactly the
	// prover-sent corner values.
	let col_emb: B128 = pi_lo
		.iter()
		.enumerate()
		.map(|(k_bit, &p)| p * beta(k_bit))
		.sum();
	let den_expected = eq_blk[0] * (tau + v00)
		+ eq_blk[1] * (tau + v10)
		+ eq_blk[2] * (tau + v01)
		+ eq_blk[3] * (tau + col_emb);
	verify_ensure!(den_expected == leaf_claim.den_eval, VerifyError::PhaseCDenominatorTransparent);

	// ---- Phase B (10K + 1 claims). ----
	let phi: B128 = IPVerifierChannel::<B128>::sample(transcript);
	let (points, evals) = phase_b_claims(dims, &ctxs, &finish_evals, &d_vals, pi_lo, m_pi);
	let combined = evaluate_univariate(&evals, phi);
	let out_b =
		binius_ip::sumcheck::verify::<B128, _>(dims.n_d, 2, combined, transcript).map_err(|e| {
			VerifyError::Sumcheck {
				phase: "phase B verify",
				source: e,
			}
		})?;
	let e_b = out_b.eval;
	let mut sigma = out_b.challenges;
	sigma.reverse(); // low-first
	let m: B128 =
		IPVerifierChannel::<B128>::recv_one(transcript).map_err(|e| VerifyError::Channel {
			phase: "phase B m",
			source: e,
		})?;
	let mut w_eq_sigma = B128::ZERO;
	let mut phi_pow = B128::ONE;
	for p in &points {
		w_eq_sigma += phi_pow * eq_point(p, &sigma);
		phi_pow *= phi;
	}
	verify_ensure!(m * w_eq_sigma == e_b, VerifyError::PhaseBFinalCheck);

	// ---- Final check: ONE merged batched opening of [M_VK, M_D]. ----
	// rho_c after m + corner values (both claims FS-bound before the opening);
	// vk_digest is FS-bound by P0.1, digest_D was observed before tau.
	let rho_c: Vec<B128> = IPVerifierChannel::<B128>::sample_many(transcript, 2);
	let claim_vk = eq_point(&[zero, zero], &rho_c) * v00
		+ eq_point(&[one, zero], &rho_c) * v10
		+ eq_point(&[zero, one], &rho_c) * v01;
	let mut vk_point = pi_lo.to_vec();
	vk_point.extend_from_slice(&rho_c);
	let vk_digest = vkm.vk_digest_typed().map_err(precondition)?;
	{
		// One Merkle channel drives the opening. Receive M_VK's commitment (oracle 0) and
		// PIN it to the audited vk_digest: the FRI queries bind only what the transcript
		// carries, so the opened root MUST equal the audited one (the table-swap / T1
		// binding). Then run the merged reduction + combined FRI over both pinned
		// commitments [M_VK, M_D].
		let mut mchan =
			VerifierMerkleTranscriptChannel::<_, C, B128, StdHashSuite>::new(&mut *transcript);
		let vk_commitment = mchan.recv_merkle_commitment(pcs.leaf_size(0), pcs.depth(0))?;
		verify_ensure!(vk_commitment.commitment.root == vk_digest, VerifyError::VkDigestPin);
		crate::merged::verify_merged_openings(
			&pcs.params,
			&mut mchan,
			&[vk_commitment, d_commitment],
			[(&vk_point, claim_vk), (&sigma, m)],
		)?;
	}

	Ok(())
}

/// Which final-check mode a discharge instance runs. STEP 1 stays available as a
/// regression/native mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DischargeMode {
	/// STEP 1: no commitments; the verifier rebuilds M_D from the CS natively.
	Step1Native,
	/// STEP 2: committed M_VK/M_D; Phase C + one merged BaseFold opening of
	/// [M_VK, M_D]; CS-free verifier.
	Step2Committed,
}

/// Verifier-side key material for [`discharge_verify_any`].
pub enum VerifierKey<'a> {
	Native(&'a ConstraintSystem),
	Committed(&'a DischargeVkm),
}

/// Mode-dispatched verification.
pub fn discharge_verify_any<C: Challenger>(
	key: VerifierKey<'_>,
	stmt: &DischargeStatement,
	transcript: &mut VerifierTranscript<C>,
) -> Result<(), VerifyError> {
	match key {
		VerifierKey::Native(cs) => crate::discharge::discharge_verify(cs, stmt, transcript),
		VerifierKey::Committed(vkm) => discharge_verify_step2(vkm, stmt, transcript),
	}
}
