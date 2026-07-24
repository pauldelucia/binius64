// Copyright 2026 The Binius Developers

//! Tests for the in-circuit Vision permutation gadgets.
//!
//! Three layers of checking:
//! 1. **Differential** — circuit outputs equal the restored native `binius-hash` Vision
//!    implementation (itself pinned by Sage-derived KATs) on random and edge inputs, for both state
//!    sizes, plus a direct differential test of the `bmul` primitive against native GHASH
//!    multiplication.
//! 2. **Adversarial** — hint-based gadgets are forgery surfaces: every hint is independently
//!    replaced by a wrong value (both via free witness wires, letting the evaluator propagate
//!    consistently, and via post-population value-vector tampering) and the constraint system must
//!    reject.
//! 3. **Counts** — `CircuitStat` constraint counts for every gadget and both full permutations are
//!    pinned exactly.
//!
//! NOTE on `verify_constraints`: the upstream helper `binius_core::verify::
//! verify_constraints` checks AND and IMUL constraints only — it silently skips
//! BMUL constraints. Adversarial tests here use [`verify_all_constraints`], which
//! adds a native-GHASH check of every `BmulConstraint`; without it, tampering a
//! value pinned only by BMUL constraints would falsely pass.

use binius_core::{ConstraintSystem, ValueVec, Word, verify::verify_constraints};
use binius_field::{BinaryField128bGhash as Ghash, Field, Random, arithmetic_traits::InvertOrZero};
use binius_frontend::{CircuitBuilder, CircuitStat};
use binius_hash::{vision_4, vision_6};
use rand::{SeedableRng, rngs::StdRng};

use super::*;
use crate::sha256::{State, sha256_compress, sha256_compress_2x_seq};

/// Verifies AND, IMUL *and* BMUL constraints against the witness.
///
/// The upstream `verify_constraints` does not check BMUL constraints; this helper
/// evaluates each `BmulConstraint` with native GHASH-field arithmetic.
fn verify_all_constraints(cs: &ConstraintSystem, witness: &ValueVec) -> Result<(), String> {
	verify_constraints(cs, witness)?;
	for (i, c) in cs.bmul_constraints.iter().enumerate() {
		let a = words_to_ghash(witness.eval_operand(&c.a_lo), witness.eval_operand(&c.a_hi));
		let b = words_to_ghash(witness.eval_operand(&c.b_lo), witness.eval_operand(&c.b_hi));
		let prod = words_to_ghash(witness.eval_operand(&c.c_lo), witness.eval_operand(&c.c_hi));
		if a * b != prod {
			return Err(format!("BMUL constraint {i} failed: a*b != c"));
		}
	}
	Ok(())
}

/// Interesting edge-case field elements: zero, one, single bits, all-ones.
fn edge_elements() -> Vec<Ghash> {
	let mut v = vec![Ghash::ZERO, Ghash::ONE, Ghash::new(u128::MAX)];
	for bit in [1, 63, 64, 127] {
		v.push(Ghash::new(1u128 << bit));
	}
	v
}

#[test]
fn test_b_maps_shared_between_sizes() {
	// The b_inv hint uses the vision_4 tables for both permutation sizes; this
	// pins that Vision-4 and Vision-6 really share identical B-maps.
	assert_eq!(vision_4::constants::B_FWD_COEFFS, vision_6::constants::B_FWD_COEFFS);
	assert_eq!(vision_4::constants::B_INV_COEFFS, vision_6::constants::B_INV_COEFFS);
}

// ---------------------------------------------------------------------------
// Differential tests
// ---------------------------------------------------------------------------

/// The `bmul` primitive itself: circuit product == native GHASH product.
#[test]
fn test_bmul_matches_native() {
	const N: usize = 64;
	let b = CircuitBuilder::new();
	let xs: Vec<GhashWire> = (0..N).map(|_| GhashWire::witness(&b)).collect();
	let ys: Vec<GhashWire> = (0..N).map(|_| GhashWire::witness(&b)).collect();
	let prods: Vec<GhashWire> = (0..N).map(|i| ghash_mul(&b, xs[i], ys[i])).collect();
	for p in &prods {
		p.force_commit(&b);
	}
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(0);
	let mut w = circuit.new_witness_filler();
	let mut expected = Vec::with_capacity(N);
	for i in 0..N {
		// Mix in edge elements among the random pairs.
		let (xv, yv) = if i < edge_elements().len() {
			(edge_elements()[i], Ghash::random(&mut rng))
		} else {
			(Ghash::random(&mut rng), Ghash::random(&mut rng))
		};
		xs[i].populate(&mut w, xv);
		ys[i].populate(&mut w, yv);
		expected.push(xv * yv);
	}
	circuit.populate_wire_witness(&mut w).unwrap();
	for i in 0..N {
		assert_eq!(prods[i].read(&w), expected[i], "bmul mismatch at pair {i}");
	}
	verify_all_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
}

#[test]
fn test_invert_or_zero_differential() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let inv = invert_or_zero(&b, x);
	inv.w.force_commit(&b);
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(1);
	let mut inputs = edge_elements();
	inputs.extend((0..32).map(|_| Ghash::random(&mut rng)));

	for xv in inputs {
		let mut w = circuit.new_witness_filler();
		x.populate(&mut w, xv);
		circuit.populate_wire_witness(&mut w).unwrap();
		assert_eq!(inv.w.read(&w), xv.invert_or_zero(), "wrong inverse for {xv:?}");
		let expected_e = if xv == Ghash::ZERO {
			Ghash::ZERO
		} else {
			Ghash::ONE
		};
		assert_eq!(inv.e.read(&w), expected_e, "wrong indicator for {xv:?}");
		verify_all_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
	}
}

#[test]
fn test_b_maps_differential() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let fwd = b_fwd_affine(&b, x);
	fwd.force_commit(&b);
	let inv = b_inv_affine_hinted(&b, x);
	inv.y.force_commit(&b);
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(2);
	let mut inputs = edge_elements();
	inputs.extend((0..32).map(|_| Ghash::random(&mut rng)));

	for xv in inputs {
		// Native references from the restored implementation.
		let mut fwd_native = [xv];
		vision_4::permutation::b_fwd_transform(&mut fwd_native);
		let mut inv_native = xv;
		vision_4::permutation::linearized_b_inv_transform_scalar(&mut inv_native);

		let mut w = circuit.new_witness_filler();
		x.populate(&mut w, xv);
		circuit.populate_wire_witness(&mut w).unwrap();
		assert_eq!(fwd.read(&w), fwd_native[0], "B_fwd mismatch for {xv:?}");
		assert_eq!(inv.y.read(&w), inv_native, "B_inv mismatch for {xv:?}");
		verify_all_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
	}
}

#[test]
fn test_mds_differential() {
	let b = CircuitBuilder::new();
	let in4: [GhashWire; 4] = std::array::from_fn(|_| GhashWire::witness(&b));
	let out4 = mds_4(&b, &in4);
	let in6: [GhashWire; 6] = std::array::from_fn(|_| GhashWire::witness(&b));
	let out6 = mds_6(&b, &in6);
	for o in out4.iter().chain(&out6) {
		o.force_commit(&b);
	}
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(3);
	for _ in 0..32 {
		let v4: [Ghash; 4] = std::array::from_fn(|_| Ghash::random(&mut rng));
		let v6: [Ghash; 6] = std::array::from_fn(|_| Ghash::random(&mut rng));
		let mut n4 = v4;
		vision_4::permutation::mds_mul(&mut n4);
		let mut n6 = v6;
		vision_6::permutation::mds_mul(&mut n6);

		let mut w = circuit.new_witness_filler();
		for i in 0..4 {
			in4[i].populate(&mut w, v4[i]);
		}
		for i in 0..6 {
			in6[i].populate(&mut w, v6[i]);
		}
		circuit.populate_wire_witness(&mut w).unwrap();
		for i in 0..4 {
			assert_eq!(out4[i].read(&w), n4[i], "mds_4 mismatch at {i}");
		}
		for i in 0..6 {
			assert_eq!(out6[i].read(&w), n6[i], "mds_6 mismatch at {i}");
		}
		verify_all_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
	}
}

/// Edge states for the full-permutation differential tests: all-zero, all-ones,
/// and single-bit states (one bit set in one element, others zero).
fn edge_states<const M: usize>() -> Vec<[Ghash; M]> {
	let mut states = vec![[Ghash::ZERO; M], [Ghash::new(u128::MAX); M]];
	for elem in 0..M {
		for bit in [0u32, 63, 64, 127] {
			let mut s = [Ghash::ZERO; M];
			s[elem] = Ghash::new(1u128 << bit);
			states.push(s);
		}
	}
	states
}

fn run_permutation_differential<const M: usize>(
	circuit_perm: impl Fn(&CircuitBuilder, [GhashWire; M]) -> [GhashWire; M],
	native_perm: impl Fn(&mut [Ghash; M]),
	seed: u64,
) {
	let b = CircuitBuilder::new();
	let input: [GhashWire; M] = std::array::from_fn(|_| GhashWire::witness(&b));
	let output = circuit_perm(&b, input);
	for o in &output {
		o.force_commit(&b);
	}
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(seed);
	let mut states = edge_states::<M>();
	states.extend((0..100).map(|_| std::array::from_fn(|_| Ghash::random(&mut rng))));

	for state in states {
		let mut expected = state;
		native_perm(&mut expected);

		let mut w = circuit.new_witness_filler();
		for i in 0..M {
			input[i].populate(&mut w, state[i]);
		}
		circuit.populate_wire_witness(&mut w).unwrap();
		for i in 0..M {
			assert_eq!(
				output[i].read(&w),
				expected[i],
				"Vision-{M} circuit/native mismatch at element {i} for input {state:?}"
			);
		}
		verify_all_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
	}
}

#[test]
fn test_vision4_differential() {
	run_permutation_differential::<4>(vision4_permutation, vision_4::permutation::permutation, 4);
}

#[test]
fn test_vision6_differential() {
	run_permutation_differential::<6>(vision6_permutation, vision_6::permutation::permutation, 5);
}

#[test]
fn test_mul_x_shift_differential() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let shifted = ghash_mul_x_shift(&b, x);
	shifted.force_commit(&b);
	let via_bmul = ghash_mul_const(&b, x, Ghash::ONE.mul_x());
	via_bmul.force_commit(&b);
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(6);
	let mut inputs = edge_elements();
	inputs.extend((0..32).map(|_| Ghash::random(&mut rng)));
	for xv in inputs {
		let mut w = circuit.new_witness_filler();
		x.populate(&mut w, xv);
		circuit.populate_wire_witness(&mut w).unwrap();
		assert_eq!(shifted.read(&w), xv.mul_x(), "shift-based mul_x mismatch for {xv:?}");
		assert_eq!(via_bmul.read(&w), xv.mul_x(), "bmul-based mul_x mismatch for {xv:?}");
		verify_all_constraints(circuit.constraint_system(), &w.into_value_vec()).unwrap();
	}
}

// ---------------------------------------------------------------------------
// Adversarial tests — hints are forgery surfaces
// ---------------------------------------------------------------------------

/// The strongest inversion attack: the "prover" supplies an arbitrary claimed
/// inverse w' via a free witness wire and every downstream value is propagated
/// *consistently* by the evaluator. The gadget's own constraints must reject every
/// w' that is not exactly invert_or_zero(x).
#[test]
fn test_invert_or_zero_rejects_wrong_witness() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let w_claim = GhashWire::witness(&b);
	// Exactly the constraints the hinted gadget emits.
	let _e = invert_or_zero_constrain(&b, x, w_claim);
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(7);
	let run = |xv: Ghash, wv: Ghash| -> Result<(), _> {
		let mut w = circuit.new_witness_filler();
		x.populate(&mut w, xv);
		w_claim.populate(&mut w, wv);
		circuit.populate_wire_witness(&mut w)
	};

	for _ in 0..16 {
		let xv = Ghash::random(&mut rng);
		// Honest witness passes.
		run(xv, xv.invert_or_zero()).expect("honest inverse must satisfy the gadget");
		// Any perturbed inverse must be rejected.
		let delta = Ghash::random(&mut rng);
		if delta != Ghash::ZERO {
			run(xv, xv.invert_or_zero() + delta).expect_err("forged inverse must be rejected");
		}
		run(xv, Ghash::ZERO).expect_err("claiming 0 for nonzero x must be rejected");
	}
	// x = 0: only w = 0 passes.
	run(Ghash::ZERO, Ghash::ZERO).expect("x=0, w=0 must satisfy the gadget");
	run(Ghash::ZERO, Ghash::ONE).expect_err("x=0 with nonzero w must be rejected");
	run(Ghash::ZERO, Ghash::random(&mut rng)).expect_err("x=0 with random w must be rejected");
}

/// Same attack on the B_inv hint: an arbitrary claimed y propagated consistently
/// must be rejected unless B_fwd(y) = x, i.e. y = B_inv(x) exactly.
#[test]
fn test_b_inv_rejects_wrong_witness() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let y_claim = GhashWire::witness(&b);
	// Exactly the constraints b_inv_affine_hinted emits for its hint output.
	let check = b_fwd_affine(&b, y_claim);
	ghash_assert_eq(&b, "b_inv_fwd_check", check, x);
	let circuit = b.build();

	let mut rng = StdRng::seed_from_u64(8);
	let run = |xv: Ghash, yv: Ghash| -> Result<(), _> {
		let mut w = circuit.new_witness_filler();
		x.populate(&mut w, xv);
		y_claim.populate(&mut w, yv);
		circuit.populate_wire_witness(&mut w)
	};

	for _ in 0..16 {
		let xv = Ghash::random(&mut rng);
		let mut y_true = xv;
		vision_4::permutation::linearized_b_inv_transform_scalar(&mut y_true);
		run(xv, y_true).expect("honest B_inv output must satisfy the gadget");
		let delta = Ghash::random(&mut rng);
		if delta != Ghash::ZERO {
			run(xv, y_true + delta).expect_err("forged B_inv output must be rejected");
		}
	}
}

/// Post-population tampering of each hint wire independently on the *hinted*
/// gadgets: flip bits in the committed value vector and require some constraint
/// (including BMUL, which upstream `verify_constraints` skips) to break.
#[test]
fn test_hint_wire_tamper_detected() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let inv = invert_or_zero(&b, x);
	let binv = b_inv_affine_hinted(&b, inv.w);
	let circuit = b.build();
	let cs = circuit.constraint_system();

	let mut rng = StdRng::seed_from_u64(9);
	let xv = Ghash::random(&mut rng);

	// Honest baseline.
	let mut w = circuit.new_witness_filler();
	x.populate(&mut w, xv);
	circuit.populate_wire_witness(&mut w).unwrap();
	verify_all_constraints(cs, &w.value_vec().clone()).unwrap();

	// Tamper each hint-derived element independently (lo and hi word of each).
	let targets = [
		("inverse hint w", inv.w),
		("indicator e", inv.e),
		("B_inv hint y", binv.y),
	];
	for (name, target) in targets {
		for flip_hi in [false, true] {
			let mut w = circuit.new_witness_filler();
			x.populate(&mut w, xv);
			circuit.populate_wire_witness(&mut w).unwrap();
			let wire = if flip_hi { target.hi } else { target.lo };
			w[wire] = Word(w[wire].as_u64() ^ 1);
			let vv = w.into_value_vec();
			assert!(
				verify_all_constraints(cs, &vv).is_err(),
				"tampering {name} ({}) must violate the constraint system",
				if flip_hi { "hi" } else { "lo" },
			);
		}
	}
}

/// The compensated attack on the full permutation: rerun with a forged first-round
/// inverse via value-vector tampering of *both* the hint wire and its indicator to
/// keep the first constraint consistent; the remaining constraints must still break.
#[test]
fn test_invert_compensated_tamper_detected() {
	let b = CircuitBuilder::new();
	let x = GhashWire::witness(&b);
	let inv = invert_or_zero(&b, x);
	let circuit = b.build();
	let cs = circuit.constraint_system();

	let mut rng = StdRng::seed_from_u64(10);
	let xv = Ghash::random(&mut rng);
	let w_forged = Ghash::random(&mut rng);
	assert_ne!(w_forged, xv.invert_or_zero());

	let mut w = circuit.new_witness_filler();
	x.populate(&mut w, xv);
	circuit.populate_wire_witness(&mut w).unwrap();
	// Forge w and recompute e = x·w consistently so the defining BMUL holds; the
	// e·x = x / e·w = w constraints must still reject.
	inv.w.populate(&mut w, w_forged);
	inv.e.populate(&mut w, xv * w_forged);
	let vv = w.into_value_vec();
	assert!(
		verify_all_constraints(cs, &vv).is_err(),
		"compensated inverse forgery must violate the constraint system"
	);
}

// ---------------------------------------------------------------------------
// Constraint counts
// ---------------------------------------------------------------------------

fn stats_of(build: impl FnOnce(&CircuitBuilder)) -> CircuitStat {
	let b = CircuitBuilder::new();
	build(&b);
	let circuit = b.build();
	CircuitStat::collect(&circuit)
}

/// Pins the exact AND/BMUL counts of every gadget and prints the probe's
/// measurement tables (`cargo test -p binius-circuits vision -- --nocapture`).
///
/// Two boundary conventions are measured, because they differ for gadgets whose
/// outputs are linear (XOR) combinations:
///
/// - **committed**: outputs are `force_commit`ed. Materializing a committed copy of a linear output
///   costs 1 AND per word, so XOR-tailed gadgets (B_fwd, MDS, the full permutations whose last step
///   is a round-constant XOR) show +2 AND per output element over their intrinsic cost.
/// - **consumed**: outputs feed a downstream BMUL (the realistic composition — e.g. the next S-box
///   inversion, or a Merkle/transcript consumer). XOR and shift terms fold into the BMUL operand
///   for free, so this measures intrinsic cost plus exactly the known consumer BMULs.
#[test]
fn test_circuit_stats() {
	// (name, stat, expected_bmul, expected_and)
	let committed: Vec<(&str, CircuitStat, usize, usize)> = vec![
		(
			"invert_or_zero",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				invert_or_zero(b, x);
			}),
			3,
			4,
		),
		(
			"b_fwd_affine",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				b_fwd_affine(b, x).force_commit(b);
			}),
			5,
			2,
		),
		(
			"b_inv_affine_hinted",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				b_inv_affine_hinted(b, x);
			}),
			5,
			2,
		),
		(
			"sbox (B_inv half)",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				let inv = invert_or_zero(b, x);
				b_inv_affine_hinted(b, inv.w).y.force_commit(b);
			}),
			8,
			6,
		),
		(
			"sbox (B_fwd half)",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				let inv = invert_or_zero(b, x);
				b_fwd_affine(b, inv.w).force_commit(b);
			}),
			8,
			6,
		),
		(
			"mds_4",
			stats_of(|b| {
				let a: [GhashWire; 4] = std::array::from_fn(|_| GhashWire::witness(b));
				for o in mds_4(b, &a) {
					o.force_commit(b);
				}
			}),
			4,
			8,
		),
		(
			"mds_6",
			stats_of(|b| {
				let a: [GhashWire; 6] = std::array::from_fn(|_| GhashWire::witness(b));
				for o in mds_6(b, &a) {
					o.force_commit(b);
				}
			}),
			12,
			12,
		),
		(
			"mul_x via const BMUL",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				ghash_mul_const(b, x, Ghash::ONE.mul_x()).force_commit(b);
			}),
			1,
			0,
		),
		(
			"mul_x via shifts",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				ghash_mul_x_shift(b, x).force_commit(b);
			}),
			0,
			3,
		),
		(
			"Vision-4 permutation",
			stats_of(|b| {
				let s: [GhashWire; 4] = std::array::from_fn(|_| GhashWire::witness(b));
				for o in vision4_permutation(b, s) {
					o.force_commit(b);
				}
			}),
			576,
			328,
		),
		(
			"Vision-6 permutation",
			stats_of(|b| {
				let s: [GhashWire; 6] = std::array::from_fn(|_| GhashWire::witness(b));
				for o in vision6_permutation(b, s) {
					o.force_commit(b);
				}
			}),
			960,
			492,
		),
	];

	// Consumed variants: outputs are chained into BMULs (`k` extra consumer BMULs
	// for `k+1` outputs; one extra for a single output), exposing the intrinsic
	// AND cost with XOR/shift tails folded into operands.
	let consume = |b: &CircuitBuilder, outs: &[GhashWire]| {
		let mut acc = outs[0];
		for o in &outs[1..] {
			acc = ghash_mul(b, acc, *o);
		}
		if outs.len() == 1 {
			let other = GhashWire::witness(b);
			acc = ghash_mul(b, acc, other);
		}
		acc.force_commit(b);
	};
	let consumed: Vec<(&str, CircuitStat, usize, usize)> = vec![
		(
			"b_fwd_affine (+1 bmul)",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				let out = b_fwd_affine(b, x);
				consume(b, &[out]);
			}),
			6,
			0,
		),
		(
			"mds_4 (+3 bmul)",
			stats_of(|b| {
				let a: [GhashWire; 4] = std::array::from_fn(|_| GhashWire::witness(b));
				consume(b, &mds_4(b, &a));
			}),
			7,
			0,
		),
		(
			"mds_6 (+5 bmul)",
			stats_of(|b| {
				let a: [GhashWire; 6] = std::array::from_fn(|_| GhashWire::witness(b));
				consume(b, &mds_6(b, &a));
			}),
			17,
			0,
		),
		(
			"mul_x shifts (+1 bmul)",
			stats_of(|b| {
				let x = GhashWire::witness(b);
				let out = ghash_mul_x_shift(b, x);
				consume(b, &[out]);
			}),
			1,
			1,
		),
		(
			"Vision-4 perm (+3 bmul)",
			stats_of(|b| {
				let s: [GhashWire; 4] = std::array::from_fn(|_| GhashWire::witness(b));
				consume(b, &vision4_permutation(b, s));
			}),
			579,
			320,
		),
		(
			"Vision-6 perm (+5 bmul)",
			stats_of(|b| {
				let s: [GhashWire; 6] = std::array::from_fn(|_| GhashWire::witness(b));
				consume(b, &vision6_permutation(b, s));
			}),
			965,
			480,
		),
	];

	println!();
	println!("committed outputs (boundary: +1 AND per committed linear word)");
	println!("| gadget                | n_bmul | n_and |");
	println!("|-----------------------|--------|-------|");
	for (name, stat, _, _) in &committed {
		println!("| {name:<21} | {:>6} | {:>5} |", stat.n_bmul_constraints, stat.n_and_constraints);
	}
	println!();
	println!("outputs consumed by BMUL (XOR/shift tails fold into operands)");
	println!("| gadget                  | n_bmul | n_and |");
	println!("|-------------------------|--------|-------|");
	for (name, stat, _, _) in &consumed {
		println!("| {name:<23} | {:>6} | {:>5} |", stat.n_bmul_constraints, stat.n_and_constraints);
	}

	for (name, stat, expected_bmul, expected_and) in committed.iter().chain(&consumed) {
		assert_eq!(
			stat.n_bmul_constraints, *expected_bmul,
			"{name}: BMUL count drifted from the pinned value"
		);
		assert_eq!(
			stat.n_and_constraints, *expected_and,
			"{name}: AND count drifted from the pinned value"
		);
	}
}

// ---------------------------------------------------------------------------
// P3a — verifier term weighting (challenger probe)
// ---------------------------------------------------------------------------

/// Per-class constraint and shifted-value-term counts of a built constraint system.
///
/// The verifier's monster sumcheck evaluates every constraint operand, and an operand
/// is a fold (XOR) of [`binius_core::ShiftedValueIndex`] terms. Verifier work per
/// constraint class is therefore proportional to the TOTAL number of such terms, not
/// to the constraint count: an AND has 3 operands, a BMUL has 6, and real gadgets
/// fold XOR-of-shift expressions of very different sizes into them. These counts are
/// the measured BMUL-vs-AND weighting for the probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct TermCounts {
	n_and: usize,
	n_bmul: usize,
	/// Total `ShiftedValueIndex` terms across all AND operands (a, b, c).
	and_terms: usize,
	/// Total `ShiftedValueIndex` terms across all BMUL operands (a/b/c × lo/hi).
	bmul_terms: usize,
}

impl TermCounts {
	fn of(cs: &ConstraintSystem) -> Self {
		let and_terms = cs
			.and_constraints
			.iter()
			.map(|c| c.a.len() + c.b.len() + c.c.len())
			.sum();
		let bmul_terms = cs
			.bmul_constraints
			.iter()
			.map(|c| {
				c.a_lo.len()
					+ c.a_hi.len() + c.b_lo.len()
					+ c.b_hi.len() + c.c_lo.len()
					+ c.c_hi.len()
			})
			.sum();
		Self {
			n_and: cs.and_constraints.len(),
			n_bmul: cs.bmul_constraints.len(),
			and_terms,
			bmul_terms,
		}
	}

	const fn total_terms(&self) -> usize {
		self.and_terms + self.bmul_terms
	}

	const fn sub(&self, other: &Self) -> Self {
		Self {
			n_and: self.n_and - other.n_and,
			n_bmul: self.n_bmul - other.n_bmul,
			and_terms: self.and_terms - other.and_terms,
			bmul_terms: self.bmul_terms - other.bmul_terms,
		}
	}

	const fn div(&self, k: usize) -> Self {
		Self {
			n_and: self.n_and / k,
			n_bmul: self.n_bmul / k,
			and_terms: self.and_terms / k,
			bmul_terms: self.bmul_terms / k,
		}
	}
}

fn counts_of(build: impl FnOnce(&CircuitBuilder)) -> TermCounts {
	let b = CircuitBuilder::new();
	build(&b);
	let circuit = b.build();
	TermCounts::of(circuit.constraint_system())
}

/// A chain of `k` Vision permutations of state size `M` (output feeds next input).
fn vision_chain<const M: usize>(
	b: &CircuitBuilder,
	k: usize,
	perm: impl Fn(&CircuitBuilder, [GhashWire; M]) -> [GhashWire; M],
) {
	let mut state: [GhashWire; M] = std::array::from_fn(|_| GhashWire::witness(b));
	for _ in 0..k {
		state = perm(b, state);
	}
	for o in &state {
		o.force_commit(b);
	}
}

/// A chain of `k` SHA-256 compressions (fresh witness message block each, chained state).
fn sha256_chain(b: &CircuitBuilder, k: usize) {
	let mut state = State::private(b);
	for _ in 0..k {
		let m: [binius_frontend::Wire; 16] = std::array::from_fn(|_| b.add_witness());
		state = sha256_compress(b, state, m);
	}
	for w in state.0 {
		b.force_commit(w);
	}
}

/// A chain of `k` two-lane-packed sequential SHA-256 compression pairs (2 blocks per
/// unit): the cheapest known in-circuit form, used by `sha256_fixed` and applicable
/// even to strictly sequential chains via the `Sha256CompressHint` lane-merge trick.
fn sha256_2x_chain(b: &CircuitBuilder, k: usize) {
	let mut state = State::private(b);
	for _ in 0..k {
		let blocks: [[binius_frontend::Wire; 16]; 2] =
			std::array::from_fn(|_| std::array::from_fn(|_| b.add_witness()));
		state = sha256_compress_2x_seq(b, state, blocks);
	}
	for w in state.0 {
		b.force_commit(w);
	}
}

/// Measures the marginal (boundary-free) per-unit cost of each hash circuit by
/// differencing chains of 4 and 8 units, and pins the resulting term weighting.
///
/// Run with `-- --nocapture` to see the table (this is the probe's P3a deliverable).
#[test]
fn test_term_counts() {
	const K0: usize = 4;
	const K1: usize = 8;
	let marginal = |f: &dyn Fn(&CircuitBuilder, usize)| {
		let small = counts_of(|b| f(b, K0));
		let large = counts_of(|b| f(b, K1));
		large.sub(&small).div(K1 - K0)
	};

	let v4 = marginal(&|b, k| vision_chain::<4>(b, k, vision4_permutation));
	let v6 = marginal(&|b, k| vision_chain::<6>(b, k, vision6_permutation));
	let sha = marginal(&|b, k| sha256_chain(b, k));
	// One 2x_seq unit = 2 compression blocks; halve to get per-block cost.
	let sha2x = marginal(&|b, k| sha256_2x_chain(b, k)).div(2);

	println!();
	println!("P3a — marginal per-unit constraint & shifted-value-term counts");
	println!("(unit = one permutation / one compression block; chained, boundary-free)");
	println!(
		"| circuit    | n_and | n_bmul | and_terms | bmul_terms | total_terms | terms/AND | terms/BMUL |"
	);
	println!(
		"|------------|-------|--------|-----------|------------|-------------|-----------|------------|"
	);
	for (name, c) in [
		("Vision-4", v4),
		("Vision-6", v6),
		("SHA-256 1x", sha),
		("SHA-256 2x", sha2x),
	] {
		let tpa = if c.n_and > 0 {
			c.and_terms as f64 / c.n_and as f64
		} else {
			0.0
		};
		let tpb = if c.n_bmul > 0 {
			c.bmul_terms as f64 / c.n_bmul as f64
		} else {
			0.0
		};
		println!(
			"| {name:<10} | {:>5} | {:>6} | {:>9} | {:>10} | {:>11} | {tpa:>9.2} | {tpb:>10.2} |",
			c.n_and,
			c.n_bmul,
			c.and_terms,
			c.bmul_terms,
			c.total_terms(),
		);
	}

	// Pin the marginal per-unit counts so the probe numbers stay reproducible.
	let pinned = [
		(
			"Vision-4",
			v4,
			TermCounts {
				n_and: 320,
				n_bmul: 576,
				and_terms: 3016,
				bmul_terms: 7312,
			},
		),
		(
			"Vision-6",
			v6,
			TermCounts {
				n_and: 480,
				n_bmul: 960,
				and_terms: 3660,
				bmul_terms: 10104,
			},
		),
		(
			"SHA-256 1x",
			sha,
			TermCounts {
				n_and: 907,
				n_bmul: 0,
				and_terms: 11629,
				bmul_terms: 0,
			},
		),
		(
			"SHA-256 2x",
			sha2x,
			TermCounts {
				n_and: 495,
				n_bmul: 0,
				and_terms: 5997,
				bmul_terms: 0,
			},
		),
	];
	for (name, got, want) in pinned {
		assert_eq!(got, want, "{name}: marginal counts drifted from the pinned values");
	}
}
