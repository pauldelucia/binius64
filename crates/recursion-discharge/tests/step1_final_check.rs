// Copyright 2026 The Binius Developers

//! Isolates the STEP-1 native final check: a lie that Phases A and B accept, caught
//! ONLY by the `M~_D(sigma) == m` rebuild. Without this test the final-check line could
//! be deleted and every other negative would still fail elsewhere.

use binius_recursion_discharge::{
	discharge::{DischargeStatement, Step1Tamper, discharge_prove_tampered, discharge_verify},
	error::VerifyError,
	leaf::LeafPipeline,
	recorder::verify_and_capture,
	synth::{synth_cs, synth_witness},
	table::extract_table,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::config::StdChallenger;

const K: usize = 2;

/// The (1,1)-block M_D lie is invisible to Phases A and B (the claim points never select
/// that block and the prover's Phase-B eval `m` is tampered-consistent), so the native
/// final check is the sole line that rejects it.
#[test]
fn md_block3_lie_caught_only_by_native_final_check() {
	let pipeline = LeafPipeline::setup(synth_cs(0)).expect("pipeline");
	let cs = pipeline.verifier.constraint_system().clone();
	let table = extract_table(&cs).expect("table");
	let mut claims = Vec::with_capacity(K);
	for i in 0..K {
		let wit = synth_witness(&cs, 5000 + i as u64).expect("witness");
		let public = LeafPipeline::public(&wit);
		let proof = pipeline.prove(&wit).expect("prove");
		claims.push(
			verify_and_capture(&pipeline.verifier, &public, proof, table.dims.arity)
				.expect("capture"),
		);
	}
	let stmt = DischargeStatement::new(&table, claims).expect("stmt");

	// Control: the same prover with no tamper verifies.
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove_tampered(&cs, &stmt, &mut pt, Step1Tamper::None).expect("honest prove");
	let mut vt = VerifierTranscript::new(StdChallenger::default(), pt.finalize());
	discharge_verify(&cs, &stmt, &mut vt).expect("control must verify");
	vt.finalize().expect("finalize");

	// Tampered: Phases A/B accept, the native final check rejects.
	let mut pt = ProverTranscript::new(StdChallenger::default());
	discharge_prove_tampered(&cs, &stmt, &mut pt, Step1Tamper::MdBlock3)
		.expect("tampered prover still produces a transcript");
	let mut vt = VerifierTranscript::new(StdChallenger::default(), pt.finalize());
	let err = discharge_verify(&cs, &stmt, &mut vt).expect_err("M_D (1,1)-block lie MUST reject");
	eprintln!("[step1-final-check] rejected: {err}");
	assert!(
		matches!(err, VerifyError::NativeFinalCheck),
		"the lie must be caught specifically by the native final check, got: {err:?}"
	);
}
