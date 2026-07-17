// Copyright 2026 The Binius Developers

//! Structural-precondition negatives: each cheap admission/metadata check must reject
//! for the RIGHT reason, plus a K=1 positive. All statement-level, no tampered provers
//! (those live in the adversarial suites); runs in milliseconds.

use binius_core::constraint_system::{ImulConstraint, ShiftedValueIndex, ValueIndex};
use binius_field::Field;
use binius_recursion_discharge::{
	B128,
	discharge::{DischargeStatement, discharge_prove, discharge_verify, statement_from_capture},
	leaf::LeafPipeline,
	recorder::verify_and_capture,
	step2::{discharge_prove_step2, discharge_verify_step2},
	synth::{synth_cs, synth_witness},
	table::{Claim, extract_table},
	vk::vkgen,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;

/// One real captured claim of the tiny synthetic shape.
fn one_claim() -> (binius_core::constraint_system::ConstraintSystem, Claim) {
	let pipeline = LeafPipeline::setup(synth_cs(0)).expect("pipeline");
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs).expect("table");
	let wit = synth_witness(&cs, 42).expect("witness");
	let public = LeafPipeline::public(&wit);
	let proof = pipeline.prove(&wit).expect("prove");
	let claim =
		verify_and_capture(&pipeline.verifier, &public, proof, table.dims.arity).expect("capture");
	(cs, claim)
}

/// P0.3: a CS carrying a REAL (non-default) IMUL constraint is rejected at table
/// extraction — the discharge admits pure-AND shapes only.
#[test]
fn andonly_rejects_real_imul_constraint() {
	let mut cs = synth_cs(0);
	cs.validate_and_prepare().expect("prepare");
	let svi = ShiftedValueIndex::plain(ValueIndex(8));
	cs.imul_constraints.push(ImulConstraint {
		a: vec![svi],
		b: vec![svi],
		hi: vec![svi],
		lo: vec![svi],
	});
	let err = extract_table(&cs).expect_err("real IMUL constraint MUST be rejected");
	assert!(format!("{err:#}").contains("P0.3"), "expected the P0.3 ANDONLY check: {err:#}");
}

/// P0.1: statement metadata lies (parity flip; wrong n_pad) are rejected by the
/// verifier's metadata-coherence check. The parity lie in particular would otherwise
/// shift every Phase-A sum by w_d — the exact `v + w_d` forgery window.
#[test]
fn statement_metadata_lies_rejected() {
	let (cs, claim) = one_claim();
	let table = extract_table(&cs).expect("table");
	let stmt = DischargeStatement::new(&table, vec![claim]).expect("stmt");

	let mut parity_lie = stmt.clone();
	parity_lie.parity = !parity_lie.parity;
	let mut vt = VerifierTranscript::new(StdChallenger::default(), Vec::new());
	let err = discharge_verify(&cs, &parity_lie, &mut vt).expect_err("parity lie MUST reject");
	assert!(format!("{err}").contains("P0.1"), "expected the P0.1 metadata check: {err}");

	let mut npad_lie = stmt;
	npad_lie.n_pad *= 2;
	let mut vt = VerifierTranscript::new(StdChallenger::default(), Vec::new());
	let err = discharge_verify(&cs, &npad_lie, &mut vt).expect_err("n_pad lie MUST reject");
	assert!(format!("{err}").contains("P0.1"), "expected the P0.1 metadata check: {err}");

	// STEP 2: both lies die against the VKM's metadata too.
	let vkm = vkgen(&table).expect("vkgen");
	for lie in [&parity_lie, &npad_lie] {
		let mut vt = VerifierTranscript::new(StdChallenger::default(), Vec::new());
		let err = discharge_verify_step2(&vkm, lie, &mut vt).expect_err("metadata lie MUST reject");
		assert!(format!("{err}").contains("P0.1"), "expected the P0.1 metadata check: {err}");
	}
}

/// P0.4: a claim with the wrong arity is rejected at statement construction, and a
/// truncated capture list is rejected by the coverage count.
#[test]
fn arity_and_coverage_rejects() {
	let (cs, claim) = one_claim();
	let table = extract_table(&cs).expect("table");

	let mut long_claim = claim.clone();
	long_claim.point.push(B128::ZERO);
	let err = DischargeStatement::new(&table, vec![long_claim])
		.expect_err("wrong-arity claim MUST be rejected");
	assert!(format!("{err:#}").contains("P0.4"), "expected the P0.4 arity check: {err:#}");

	let err = statement_from_capture(&table, vec![claim], 2)
		.expect_err("a truncated capture list MUST be rejected");
	assert!(format!("{err:#}").contains("P0.4"), "expected the P0.4 coverage check: {err:#}");
}

/// The (G) decomposition requires lambda_and outside {0, 1}; a claim with lambda_and
/// in that set aborts at claim intake (verifier side).
#[test]
fn lambda_and_in_01_aborts() {
	let (cs, claim) = one_claim();
	let table = extract_table(&cs).expect("table");
	for bad in [B128::ZERO, B128::ONE] {
		let mut bad_claim = claim.clone();
		bad_claim.point[1] = bad; // index 1 is lambda_and
		let stmt = DischargeStatement::new(&table, vec![bad_claim]).expect("arity is fine");
		let mut vt = VerifierTranscript::new(StdChallenger::default(), Vec::new());
		let err = discharge_verify(&cs, &stmt, &mut vt).expect_err("lambda in {0,1} MUST abort");
		assert!(format!("{err}").contains("lambda_and"), "expected the lambda abort: {err}");
	}
}

/// K=1 positive: a single-claim batch closes end-to-end in both modes (the smallest
/// batch exercises the mu-batching and W_eq paths at their degenerate size).
#[test]
fn k1_positive_both_steps() {
	let (cs, claim) = one_claim();
	let table = extract_table(&cs).expect("table");
	let stmt = statement_from_capture(&table, vec![claim], 1).expect("coverage");

	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove(&cs, &stmt, &mut pt).expect("step1 prove");
	let mut vt = VerifierTranscript::new(StdChallenger::default(), pt.finalize());
	discharge_verify(&cs, &stmt, &mut vt).expect("step1 verify at K=1");
	vt.finalize().expect("finalize");

	let vkm = vkgen(&table).expect("vkgen");
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove_step2(&cs, &vkm, &stmt, &mut pt).expect("step2 prove");
	let mut vt = VerifierTranscript::new(StdChallenger::default(), pt.finalize());
	discharge_verify_step2(&vkm, &stmt, &mut vt).expect("step2 verify at K=1");
	vt.finalize().expect("finalize");
}
