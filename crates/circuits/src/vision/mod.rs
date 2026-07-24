// Copyright 2026 The Binius Developers

//! In-circuit Vision permutation gadgets over GF(2^128) (the GHASH field).
//!
//! Implements the Vision-4 (M=4 state) and Vision-6 (M=6 state) permutations from
//! `binius-hash` as circuit gadgets, using the native `BMUL` constraint
//! ([`CircuitBuilder::bmul`]) for field multiplication. A GF(2^128) element is carried
//! by a pair of 64-bit wires ([`GhashWire`]): `lo` holds the coefficients of
//! `1, X, …, X^63` and `hi` those of `X^64, …, X^127` — the same convention as `bmul`
//! and as the little-endian `u128` representation of `BinaryField128bGhash`.
//!
//! # Structure (mirrors `binius_hash::vision_4::permutation`)
//!
//! Initial whitening (round-constant add), then 8 rounds of two half-rounds each.
//! A half-round is: S-box layer → circulant MDS → round-constant add. The S-box is
//! `invert_or_zero` (x ↦ x⁻¹, 0 ↦ 0) followed by a GF(2)-linearized affine "B-map":
//! the first half-round applies `B_inv` (a 129-coefficient linearized polynomial), the
//! second `B_fwd` (`B₀ + B₁·y + B₂·y² + B₃·y⁴`). `B_inv` and `B_fwd` are mutually
//! inverse affine bijections.
//!
//! # Arithmetization
//!
//! Nondeterministic witness values ("hints") keep the expensive maps out of the
//! constraint system; every hint is pinned down by cheap forward constraints:
//!
//! - **`invert_or_zero`** — hint `w` (claimed x⁻¹, or 0 for x = 0), then constrain `e := x·w`, `e·x
//!   = x`, `e·w = w`. 3 BMUL + 4 AND (two 2-word equality asserts). See
//!   [`invert_or_zero_constrain`] for the soundness argument.
//! - **`B_inv` half-rounds** — never arithmetize the 129-coefficient polynomial. Hint the output
//!   `y` and verify with the forward map: `B_fwd(y) = w`. Since `B_fwd` is a bijection, exactly one
//!   `y` satisfies this. 5 BMUL + 2 AND.
//! - **`B_fwd` half-rounds** — computed directly: 2 squarings (a squaring is one BMUL of a value
//!   with itself) + 3 constant multiplications + free XORs. 5 BMUL.
//! - **MDS + constants** — the circulant MDS entries are small powers of X. Because
//!   `mul_x`/`mul_inv_x` are GF(2)-linear, `mul_x(a ⊕ b) = mul_x(a) ⊕ mul_x(b)`, so each MDS
//!   application needs only `mul_x(aᵢ)` (and `mul_inv_x(aᵢ)` for Vision-6) once per state element,
//!   combined by XOR. v1 implements these constant multiplications as BMUL-with-constant: 4 BMUL
//!   per Vision-4 MDS, 12 per Vision-6 MDS. [`ghash_mul_x_shift`] prototypes the shift/XOR
//!   alternative. Round-constant XORs are linear (no AND/BMUL).
//!
//! # Cost summary (exact, pinned by `tests::test_circuit_stats`)
//!
//! | gadget                    | BMUL | of which const-mult | AND |
//! |---------------------------|------|---------------------|-----|
//! | `invert_or_zero`          | 3    | 0                   | 4   |
//! | `b_fwd_affine`            | 5    | 3                   | 0   |
//! | `b_inv_affine_hinted`     | 5    | 3                   | 2   |
//! | S-box (B_inv half-round)  | 8    | 3                   | 6   |
//! | S-box (B_fwd half-round)  | 8    | 3                   | 4   |
//! | `mds_4`                   | 4    | 4                   | 0   |
//! | `mds_6`                   | 12   | 12                  | 0   |
//! | Vision-4 permutation      | 576  | 256                 | 320 |
//! | Vision-6 permutation      | 960  | 480                 | 480 |
//!
//! Costs are *intrinsic*: they assume the gadget's outputs are consumed by
//! downstream BMUL/AND operands, where trailing XOR (and shift) terms fold in for
//! free. Committing a linear output to the value vector (e.g. `force_commit` on an
//! MDS or round-constant XOR result with no consumer) additionally materializes it
//! at 1 AND per word — the stats test pins both boundary conventions.

#[cfg(test)]
mod tests;

use binius_core::word::Word;
use binius_field::{BinaryField128bGhash as Ghash, Field, arithmetic_traits::InvertOrZero};
use binius_frontend::{CircuitBuilder, Wire, hints::Hint};
use binius_hash::{vision_4, vision_6};

/// A GF(2^128) GHASH-field element carried by two 64-bit wires.
///
/// `lo` holds the coefficients of `1, X, …, X^63`; `hi` holds `X^64, …, X^127`.
/// Every 128-bit string is a valid field element, so no canonicity constraints are
/// ever needed.
#[derive(Clone, Copy, Debug)]
pub struct GhashWire {
	pub lo: Wire,
	pub hi: Wire,
}

/// Splits a native field element into its `(lo, hi)` word representation.
pub fn ghash_to_words(x: Ghash) -> (Word, Word) {
	let v: u128 = x.into();
	(Word(v as u64), Word((v >> 64) as u64))
}

/// Reassembles a native field element from its `(lo, hi)` word representation.
pub const fn words_to_ghash(lo: Word, hi: Word) -> Ghash {
	Ghash::new(((hi.as_u64() as u128) << 64) | (lo.as_u64() as u128))
}

impl GhashWire {
	/// Adds a constant field element to the circuit.
	pub fn constant(b: &CircuitBuilder, val: Ghash) -> Self {
		let (lo, hi) = ghash_to_words(val);
		Self {
			lo: b.add_constant(lo),
			hi: b.add_constant(hi),
		}
	}

	/// Adds a private-witness field element to the circuit.
	pub fn witness(b: &CircuitBuilder) -> Self {
		Self {
			lo: b.add_witness(),
			hi: b.add_witness(),
		}
	}

	/// Populates this element's wires in the witness filler.
	pub fn populate(&self, w: &mut binius_frontend::WitnessFiller, val: Ghash) {
		let (lo, hi) = ghash_to_words(val);
		w[self.lo] = lo;
		w[self.hi] = hi;
	}

	/// Reads this element's value back from the witness filler.
	pub fn read(&self, w: &binius_frontend::WitnessFiller) -> Ghash {
		words_to_ghash(w[self.lo], w[self.hi])
	}

	/// Marks both wires as forcibly committed, keeping them (and the gadgets
	/// producing them) live through dead-code elimination even when no further
	/// constraint consumes them. Callers that feed the element into follow-up
	/// constraints (the normal case) do not need this.
	pub fn force_commit(&self, b: &CircuitBuilder) {
		b.force_commit(self.lo);
		b.force_commit(self.hi);
	}
}

/// Field multiplication: 1 BMUL constraint.
pub fn ghash_mul(b: &CircuitBuilder, x: GhashWire, y: GhashWire) -> GhashWire {
	let (lo, hi) = b.bmul(x.lo, x.hi, y.lo, y.hi);
	GhashWire { lo, hi }
}

/// Multiplication by a compile-time constant: 1 BMUL constraint (with constant operand).
///
/// This is the v1 costing for all MDS/B-map constant multiplications; counted
/// separately in the stats because it is the optimization headroom (a fixed sparse
/// GF(2)-linear map could instead be encoded with shifts and XORs, see
/// [`ghash_mul_x_shift`]).
pub fn ghash_mul_const(b: &CircuitBuilder, x: GhashWire, c: Ghash) -> GhashWire {
	ghash_mul(b, x, GhashWire::constant(b, c))
}

/// Field squaring: 1 BMUL constraint (both operands the same wires).
pub fn ghash_square(b: &CircuitBuilder, x: GhashWire) -> GhashWire {
	ghash_mul(b, x, x)
}

/// Field addition (XOR): linear, no AND/BMUL constraints.
pub fn ghash_xor(b: &CircuitBuilder, x: GhashWire, y: GhashWire) -> GhashWire {
	GhashWire {
		lo: b.bxor(x.lo, y.lo),
		hi: b.bxor(x.hi, y.hi),
	}
}

/// Multi-way field addition (XOR): linear, no AND/BMUL constraints.
pub fn ghash_xor_multi(b: &CircuitBuilder, xs: &[GhashWire]) -> GhashWire {
	let los: Vec<Wire> = xs.iter().map(|x| x.lo).collect();
	let his: Vec<Wire> = xs.iter().map(|x| x.hi).collect();
	GhashWire {
		lo: b.bxor_multi(&los),
		hi: b.bxor_multi(&his),
	}
}

/// Asserts equality of two field elements: 2 AND constraints (one per word).
pub fn ghash_assert_eq(b: &CircuitBuilder, name: &str, x: GhashWire, y: GhashWire) {
	b.assert_eq(format!("{name}_lo"), x.lo, y.lo);
	b.assert_eq(format!("{name}_hi"), x.hi, y.hi);
}

/// Multiplication by X via word shifts instead of a BMUL constraint.
///
/// `X·a` shifts the 128-bit coefficient vector up by one and, when the top bit
/// (`X^127` coefficient) was set, XORs in the GHASH reduction `0x87`
/// (`X^7 + X^2 + X + 1`). In the word model:
///
/// ```text
/// hi' = (hi << 1) | (lo >> 63)
/// lo' = (lo << 1) ^ (0x87 & sar(hi, 63))    // sar broadcasts the top bit
/// ```
///
/// Cost (measured): 0 BMUL and just **1 AND** (the reduction mask) when the result
/// is consumed by a downstream BMUL — the shifts and XORs fold into constraint
/// operands for free (3 AND when the result is force-committed instead). Versus
/// 1 BMUL for [`ghash_mul_const`] with `X`: this is the optimization headroom on
/// the 256 (Vision-4) / 480 (Vision-6) constant-mult BMULs per permutation.
/// Prototyped for the probe's cost comparison; the permutation gadgets use the
/// BMUL form (v1).
pub fn ghash_mul_x_shift(b: &CircuitBuilder, a: GhashWire) -> GhashWire {
	let hi_shifted = b.shl(a.hi, 1);
	let carry = b.shr(a.lo, 63);
	let hi = b.bxor(hi_shifted, carry);

	let lo_shifted = b.shl(a.lo, 1);
	// All-ones iff the X^127 coefficient is set.
	let top_broadcast = b.sar(a.hi, 63);
	let reduction = b.band(top_broadcast, b.add_constant(Word(0x87)));
	let lo = b.bxor(lo_shifted, reduction);

	GhashWire { lo, hi }
}

// ---------------------------------------------------------------------------
// Hints
// ---------------------------------------------------------------------------

/// Witness hint computing `x⁻¹` (or 0 for x = 0) in the GHASH field.
///
/// Prover-side only; the result is constrained by [`invert_or_zero_constrain`].
pub struct GhashInvertHint;

impl Hint for GhashInvertHint {
	const NAME: &'static str = "binius.vision.ghash_invert_or_zero";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2, 2)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let x = words_to_ghash(inputs[0], inputs[1]);
		let w = x.invert_or_zero();
		let (lo, hi) = ghash_to_words(w);
		outputs[0] = lo;
		outputs[1] = hi;
	}
}

/// Witness hint computing the affine `B_inv` map (129-coefficient linearized
/// polynomial) of the Vision S-box.
///
/// Prover-side only; the result `y` is constrained by verifying the cheap forward
/// map `B_fwd(y) = input` (see [`b_inv_affine_hinted`]). Vision-4 and Vision-6
/// share the same B-maps (pinned by a test), so this single hint serves both.
pub struct VisionBInvHint;

impl Hint for VisionBInvHint {
	const NAME: &'static str = "binius.vision.b_inv_affine";

	fn shape(&self, _dimensions: &[usize]) -> (usize, usize) {
		(2, 2)
	}

	fn execute(&self, _dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]) {
		let mut y = words_to_ghash(inputs[0], inputs[1]);
		// Table-driven evaluation of the affine B_inv map from the native impl.
		vision_4::permutation::linearized_b_inv_transform_scalar(&mut y);
		let (lo, hi) = ghash_to_words(y);
		outputs[0] = lo;
		outputs[1] = hi;
	}
}

// ---------------------------------------------------------------------------
// invert_or_zero gadget
// ---------------------------------------------------------------------------

/// Wire handles of one `invert_or_zero` instance, exposed for adversarial tests.
pub struct InvertOrZeroWires {
	/// The claimed inverse (hint output; equals `x⁻¹`, or 0 when x = 0).
	pub w: GhashWire,
	/// `e := x·w` (1 when x ≠ 0, else 0).
	pub e: GhashWire,
}

/// Emits the `invert_or_zero` *constraints* for a given `(x, w)` pair and returns
/// `e := x·w`. Cost: 3 BMUL + 4 AND.
///
/// Constraints: `e = x·w` (the defining BMUL), `e·x = x`, `e·w = w`.
///
/// # Soundness
///
/// The prover chooses `w` freely; `e` is *defined* by the first BMUL, so only the
/// two equality constraints restrict `w`:
///
/// - If `x ≠ 0`: `e·x = x` gives `x·w·x = x`; multiplying by `x⁻¹` (which exists) twice yields `w =
///   x⁻¹`, hence `e = 1`, and `e·w = w` holds automatically.
/// - If `x = 0`: `e = 0·w = 0`, so `e·x = x` reads `0 = 0` (no restriction), and `e·w = w` reads `0
///   = w`, forcing `w = 0`.
///
/// In both cases `(w, e)` is uniquely `(x⁻¹, 1)` or `(0, 0)` — exactly the
/// `invert_or_zero` semantics of the native S-box. There is no assumption that
/// S-box inputs are nonzero.
pub fn invert_or_zero_constrain(b: &CircuitBuilder, x: GhashWire, w: GhashWire) -> GhashWire {
	let e = ghash_mul(b, x, w); // e := x·w        (1 BMUL)
	let ex = ghash_mul(b, e, x); // (1 BMUL)
	ghash_assert_eq(b, "invert_or_zero_ex", ex, x); // e·x = x  (2 AND)
	let ew = ghash_mul(b, e, w); // (1 BMUL)
	ghash_assert_eq(b, "invert_or_zero_ew", ew, w); // e·w = w  (2 AND)
	e
}

/// `invert_or_zero` gadget: returns `x⁻¹` (or 0 for x = 0), witnessed by hint and
/// pinned by [`invert_or_zero_constrain`]. Cost: 3 BMUL + 4 AND.
pub fn invert_or_zero(b: &CircuitBuilder, x: GhashWire) -> InvertOrZeroWires {
	let out = b.call_hint(GhashInvertHint, &[], &[x.lo, x.hi]);
	let w = GhashWire {
		lo: out[0],
		hi: out[1],
	};
	let e = invert_or_zero_constrain(b, x, w);
	InvertOrZeroWires { w, e }
}

// ---------------------------------------------------------------------------
// B-maps
// ---------------------------------------------------------------------------

/// Returns the shared Vision B_fwd coefficients `[B₀, B₁, B₂, B₃]`
/// (`B_fwd(y) = B₀ + B₁·y + B₂·y² + B₃·y⁴`).
fn b_fwd_coeffs() -> [Ghash; 4] {
	// Vision-4 and Vision-6 share identical B-maps; pinned by
	// `tests::test_b_maps_shared_between_sizes`.
	vision_4::constants::B_FWD_COEFFS
}

/// Direct forward affine B-map: `B₀ + B₁·y + B₂·y² + B₃·y⁴`.
///
/// Cost: 5 BMUL (2 squarings + 3 constant mults) + free XORs.
pub fn b_fwd_affine(b: &CircuitBuilder, y: GhashWire) -> GhashWire {
	let [c0, c1, c2, c3] = b_fwd_coeffs();
	let y2 = ghash_square(b, y); // y²   (1 BMUL)
	let y4 = ghash_square(b, y2); // y⁴   (1 BMUL)
	let t1 = ghash_mul_const(b, y, c1); // (1 BMUL, const)
	let t2 = ghash_mul_const(b, y2, c2); // (1 BMUL, const)
	let t3 = ghash_mul_const(b, y4, c3); // (1 BMUL, const)
	let c0w = GhashWire::constant(b, c0);
	ghash_xor_multi(b, &[c0w, t1, t2, t3])
}

/// Wire handles of one hinted `B_inv` instance, exposed for adversarial tests.
pub struct BInvWires {
	/// The claimed `B_inv(x)` (hint output).
	pub y: GhashWire,
}

/// Hinted inverse affine B-map: returns `y = B_inv(x)`, verified via the forward
/// map. Cost: 5 BMUL + 2 AND.
///
/// # Soundness
///
/// The constraint is `B_fwd(y) = x`. `B_fwd` is an affine bijection on GF(2^128)
/// (its linear part is an invertible linearized polynomial — the native KATs pin
/// `B_inv(B_fwd(z)) = z`), so for each `x` exactly one `y` satisfies it, namely
/// `B_inv(x)`. The 129-coefficient inverse polynomial never enters the circuit.
pub fn b_inv_affine_hinted(b: &CircuitBuilder, x: GhashWire) -> BInvWires {
	let out = b.call_hint(VisionBInvHint, &[], &[x.lo, x.hi]);
	let y = GhashWire {
		lo: out[0],
		hi: out[1],
	};
	let check = b_fwd_affine(b, y); // 5 BMUL
	ghash_assert_eq(b, "b_inv_fwd_check", check, x); // 2 AND
	BInvWires { y }
}

// ---------------------------------------------------------------------------
// S-box layers
// ---------------------------------------------------------------------------

/// Which affine B-map a half-round applies after inversion.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BMap {
	/// First half-round: `B_inv` (hinted, forward-verified).
	Inv,
	/// Second half-round: `B_fwd` (computed directly).
	Fwd,
}

/// Applies the Vision S-box (`invert_or_zero` then a B-map) to every state element.
fn sbox_layer<const M: usize>(
	b: &CircuitBuilder,
	state: &[GhashWire; M],
	bmap: BMap,
) -> [GhashWire; M] {
	std::array::from_fn(|i| {
		let inv = invert_or_zero(b, state[i]);
		match bmap {
			BMap::Inv => b_inv_affine_hinted(b, inv.w).y,
			BMap::Fwd => b_fwd_affine(b, inv.w),
		}
	})
}

// ---------------------------------------------------------------------------
// MDS layers
// ---------------------------------------------------------------------------

/// Vision-4 circulant MDS (`circ(2,3,1,1)`), mirroring the native
/// `vision_4::permutation::mds_mul`:
///
/// ```text
/// out[i] = a[i] ⊕ sum ⊕ X·(a[i] ⊕ a[i+1 mod 4]),   sum = a0 ⊕ a1 ⊕ a2 ⊕ a3
/// ```
///
/// `mul_x` is GF(2)-linear, so `X·(aᵢ ⊕ aⱼ) = X·aᵢ ⊕ X·aⱼ`: only the four `X·aᵢ`
/// products are needed. Cost: 4 BMUL (all constant mults) + free XORs.
pub fn mds_4(b: &CircuitBuilder, a: &[GhashWire; 4]) -> [GhashWire; 4] {
	let x = Ghash::ONE.mul_x();
	let mx: [GhashWire; 4] = std::array::from_fn(|i| ghash_mul_const(b, a[i], x));
	let sum = ghash_xor_multi(b, a);
	std::array::from_fn(|i| {
		let j = (i + 1) % 4;
		ghash_xor_multi(b, &[a[i], sum, mx[i], mx[j]])
	})
}

/// Vision-6 circulant MDS (first row `[X⁻¹, X⁻¹, X, 1, X, X+1]`), mirroring the
/// native `vision_6::permutation::mds_mul` row by row. By GF(2)-linearity of
/// `mul_x`/`mul_inv_x`, only `X·aᵢ` and `X⁻¹·aᵢ` per element are needed.
/// Cost: 12 BMUL (all constant mults) + free XORs.
pub fn mds_6(b: &CircuitBuilder, a: &[GhashWire; 6]) -> [GhashWire; 6] {
	let x = Ghash::ONE.mul_x();
	let x_inv = Ghash::ONE.mul_inv_x();
	let mx: [GhashWire; 6] = std::array::from_fn(|i| ghash_mul_const(b, a[i], x));
	let mi: [GhashWire; 6] = std::array::from_fn(|i| ghash_mul_const(b, a[i], x_inv));

	[
		// (a0+a1)·X⁻¹ + (a2+a4)·X + a3 + a5·(X+1)
		ghash_xor_multi(b, &[mi[0], mi[1], mx[2], mx[4], a[3], mx[5], a[5]]),
		// a0·(X+1) + (a1+a2)·X⁻¹ + a3·X + a4 + a5·X
		ghash_xor_multi(b, &[mx[0], a[0], mi[1], mi[2], mx[3], a[4], mx[5]]),
		// a0·X + a1·(X+1) + (a2+a3)·X⁻¹ + a4·X + a5
		ghash_xor_multi(b, &[mx[0], mx[1], a[1], mi[2], mi[3], mx[4], a[5]]),
		// a0 + a1·X + a2·(X+1) + (a3+a4)·X⁻¹ + a5·X
		ghash_xor_multi(b, &[a[0], mx[1], mx[2], a[2], mi[3], mi[4], mx[5]]),
		// a0·X + a1 + a2·X + a3·(X+1) + (a4+a5)·X⁻¹
		ghash_xor_multi(b, &[mx[0], a[1], mx[2], mx[3], a[3], mi[4], mi[5]]),
		// (a0+a5)·X⁻¹ + a1·X + a2 + a3·X + a4·(X+1)
		ghash_xor_multi(b, &[mi[0], mi[5], mx[1], a[2], mx[3], mx[4], a[4]]),
	]
}

// ---------------------------------------------------------------------------
// Full permutations
// ---------------------------------------------------------------------------

/// XORs per-element round constants into the state (linear; no AND/BMUL).
fn add_round_constants<const M: usize>(
	b: &CircuitBuilder,
	state: &[GhashWire; M],
	rc: &[Ghash; M],
) -> [GhashWire; M] {
	std::array::from_fn(|i| ghash_xor(b, state[i], GhashWire::constant(b, rc[i])))
}

/// Shared round schedule: whitening, then 8 rounds × (B_inv half-round, B_fwd
/// half-round), mirroring the native `permutation`.
fn vision_permutation<const M: usize>(
	b: &CircuitBuilder,
	state: [GhashWire; M],
	round_constants: &[[Ghash; M]],
	mds: impl Fn(&CircuitBuilder, &[GhashWire; M]) -> [GhashWire; M],
	num_rounds: usize,
) -> [GhashWire; M] {
	assert_eq!(round_constants.len(), 1 + 2 * num_rounds);
	let mut state = add_round_constants(b, &state, &round_constants[0]);
	for r in 0..num_rounds {
		// First half-round: S-box (B_inv) → MDS → constants.
		let s = sbox_layer(b, &state, BMap::Inv);
		let m = mds(b, &s);
		state = add_round_constants(b, &m, &round_constants[1 + 2 * r]);
		// Second half-round: S-box (B_fwd) → MDS → constants.
		let s = sbox_layer(b, &state, BMap::Fwd);
		let m = mds(b, &s);
		state = add_round_constants(b, &m, &round_constants[2 + 2 * r]);
	}
	state
}

/// The full Vision-4 permutation on a 4-element GF(2^128) state.
///
/// Byte-identical to `binius_hash::vision_4::permutation::permutation` (pinned by
/// differential tests against the restored native implementation and its KATs).
///
/// Cost: 576 BMUL (of which 256 constant mults) + 320 AND.
pub fn vision4_permutation(b: &CircuitBuilder, state: [GhashWire; 4]) -> [GhashWire; 4] {
	vision_permutation(
		b,
		state,
		&vision_4::constants::ROUND_CONSTANTS,
		mds_4,
		vision_4::constants::NUM_ROUNDS,
	)
}

/// The full Vision-6 permutation on a 6-element GF(2^128) state.
///
/// Byte-identical to `binius_hash::vision_6::permutation::permutation` (pinned by
/// differential tests against the restored native implementation and its KATs).
///
/// Cost: 960 BMUL (of which 480 constant mults) + 480 AND.
pub fn vision6_permutation(b: &CircuitBuilder, state: [GhashWire; 6]) -> [GhashWire; 6] {
	vision_permutation(
		b,
		state,
		&vision_6::constants::ROUND_CONSTANTS,
		mds_6,
		vision_6::constants::NUM_ROUNDS,
	)
}
