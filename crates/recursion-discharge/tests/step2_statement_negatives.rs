// Copyright 2026 The Binius Developers

//! STEP-2 statement-level negatives: forged monster values against the COMMITTED-table
//! verifier (the STEP-1 suites cover the native verifier; these close the same holes on
//! the CS-free path).
//!
//! The second test is the sharpest STEP-2 adversary: a "consistent lie" — claims
//! recomputed against a TAMPERED term table, so Phases A and B are fully
//! self-consistent and only the committed verification-key binding (the audited
//! `vk_digest` pin at the merged opening) can reject. This is what makes the committed
//! table a verification KEY rather than advisory data.

use binius_field::Field;
use binius_recursion_discharge::{
	B128,
	discharge::DischargeStatement,
	step2::{
		Step2Tamper, discharge_prove_step2, discharge_prove_step2_on_table, discharge_verify_step2,
	},
	synth::{prepared_synth_table, synth_claims},
	vk::vkgen,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;

const K: usize = 2;
const N_Y_LOG: usize = 12;

/// A corrupted claim value must be rejected on the STEP-2 path: either the honest
/// prover fails to close Phase A against the wrong statement sum, or the verifier
/// rejects the transcript.
#[test]
fn step2_corrupted_value_rejected() -> anyhow::Result<()> {
	let (cs, table) = prepared_synth_table(8, N_Y_LOG, 0)?;
	let vkm = vkgen(&table)?;
	let mut claims = synth_claims(&table.dims, &table, 11_000, K)?;
	claims[0].value += B128::ONE;
	let stmt = DischargeStatement::new(&table, claims)?;

	let mut pt = ProverTranscript::new(StdChallenger::default());
	let prove_res = discharge_prove_step2(&cs, &vkm, &stmt, &mut pt);
	if prove_res.is_ok() {
		let mut vt = VerifierTranscript::new(StdChallenger::default(), pt.finalize());
		let verify_res = discharge_verify_step2(&vkm, &stmt, &mut vt)
			.map_err(anyhow::Error::from)
			.and_then(|_| vt.finalize().map_err(|e| anyhow::anyhow!("finalize: {e}")));
		assert!(verify_res.is_err(), "corrupted claim value MUST be rejected on STEP 2");
	}
	// Either the prover fails to close or the verifier rejects — both are correct.
	Ok(())
}

/// The consistent-lie adversary: tamper ONE term of the table (a different witness-word
/// index for term 0), recompute the claims against the tampered table, and run the full
/// honest STEP-2 machinery over the tampered table — while committing the HONEST M_VK so
/// its digest matches the audited vk_digest. Phases A, B, and even Phase C are all
/// self-consistent with the tampered table by construction (the prover's corner values
/// and fracadd tree are the tampered table's). The lie surfaces only at the merged
/// opening: the verifier's M_VK claim is built from the tampered corner values, but the
/// opened M_VK is the honest table pinned to the audited vk_digest, so its evaluation
/// disagrees and the opening rejects.
#[test]
fn step2_consistent_table_lie_rejected() -> anyhow::Result<()> {
	let (_cs, table) = prepared_synth_table(8, N_Y_LOG, 0)?;
	let vkm = vkgen(&table)?;

	// Tampered twin: one term's witness-word index shifted by one (stays in range, so
	// all dims are unchanged and the statement passes every metadata check).
	let mut tampered = table.clone();
	let y0 = tampered.terms[0].y;
	tampered.terms[0].y = if (y0 as usize) + 1 < tampered.dims.combined_len {
		y0 + 1
	} else {
		y0 - 1
	};
	assert_eq!(tampered.dims, table.dims, "the lie must not change the shape");

	// Claims synthesized against the TAMPERED table: v = monster_{tampered}(c). If the
	// discharge accepted these, it would certify monster values of a table that is NOT
	// the audited one.
	let claims = synth_claims(&tampered.dims, &tampered, 12_000, K)?;
	// The tampered table only mutates a term, so it keeps the honest cs_digest; the
	// statement already carries the audited metadata. (The explicit pin below documents
	// the adversary's obligation to match P0.2, even though it is a no-op here.)
	let mut stmt = DischargeStatement::new(&tampered, claims)?;
	stmt.cs_digest = vkm.cs_digest;

	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove_step2_on_table(&tampered, &table, &vkm, &stmt, &mut pt, Step2Tamper::None)
		.expect("the adversarial prover completes its transcript");
	let mut vt = VerifierTranscript::new(StdChallenger::default(), pt.finalize());
	let err = discharge_verify_step2(&vkm, &stmt, &mut vt)
		.expect_err("a consistent table lie MUST be rejected");
	let msg = format!("{err}");
	eprintln!("[consistent-lie] rejected: {msg}");
	// The whole transcript is self-consistent with the tampered table (Phases A, B, and C
	// all accept); only the merged opening against the honest, vk_digest-pinned M_VK
	// disagrees.
	assert!(
		msg.contains("merged"),
		"expected the merged-opening rejection against the pinned M_VK, got: {msg}"
	);
	Ok(())
}
