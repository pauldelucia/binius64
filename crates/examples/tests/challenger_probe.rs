// Copyright 2026 The Binius Developers

//! Challenger-probe instrumentation (P3b/P4 of the Vision recursion gate, issue #1885).
//!
//! Measures, for a REAL `Verifier::verify` run:
//!
//! - total Fiat-Shamir challenger traffic (bytes observed / sampled per phase) and the exact number
//!   of SHA-256 compression blocks the `HasherChallenger<Sha256>` consumes for it;
//! - Merkle verification traffic: leaf-hash invocations (with message lengths and their exact
//!   SHA-256 block counts) and 2-to-1 compression invocations;
//!
//! then maps the same traffic onto a Vision transcript (Merkle 2-to-1 -> one Vision-4
//! permutation; challenger + leaf hashing -> Vision-6 sponge at rate 64 bytes/permutation)
//! and prices both in in-circuit constraint terms measured by
//! `binius-circuits/src/vision/tests.rs::test_term_counts` (P3a).
//!
//! All counting types are byte-transparent wrappers: the instrumented verifier consumes a
//! proof produced by the STANDARD `Sha256HashSuite` prover + `StdChallenger`, and the fact
//! that verification passes is itself the equivalence check (any divergence in challenge
//! bytes would reject). An additional direct equivalence test pins the counting challenger
//! against `HasherChallenger<Sha256>` byte for byte.
//!
//! The heavy probes are `#[ignore]`d; run them one at a time (16GB laptop budget):
//!
//! ```sh
//! cargo test -p binius-examples --release --test challenger_probe -- --ignored \
//!     --test-threads=1 --nocapture
//! ```

use std::{
	array,
	collections::BTreeMap,
	marker::PhantomData,
	mem,
	sync::{
		Mutex,
		atomic::{AtomicU64, Ordering::Relaxed},
	},
	time::Instant,
};

use binius_circuits::{
	sha256::sha256_fixed,
	vision::{GhashWire, vision4_permutation, vision6_permutation},
};
use binius_core::constraint_system::ConstraintSystem;
use binius_examples::circuits::utils::pack_bytes_u32words;
use binius_field::{BinaryField128bGhash as Ghash, Random};
use binius_frontend::{CircuitBuilder, Wire};
use binius_hash::{
	CompressionFunction, ParallelCompressionAdaptor, ParallelDigestAdapter,
	binary_merkle_tree::HashSuite,
	parallel_compression::ParallelPseudoCompression,
	sha256::{ParallelSha256Compression, Sha256Compression, Sha256HashSuite},
	vision_4::{compression::VisionCompression, parallel_compression::VisionParallelCompression},
};
use binius_prover::{OptimalPackedB128, Prover};
use binius_transcript::fiat_shamir::{Challenger, HasherChallenger};
use binius_verifier::{
	Verifier,
	config::StdChallenger,
	transcript::{ProverTranscript, VerifierTranscript},
};
use bytes::{Buf, BufMut, buf::UninitSlice};
use digest::{
	Digest, FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update,
	block_api::{Block, BlockSizeUser},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use sha2::Sha256;

const LOG_INV_RATE: usize = 1;

// ---------------------------------------------------------------------------
// P3a cost constants (measured & pinned by binius-circuits vision::tests::test_term_counts)
// ---------------------------------------------------------------------------

/// (n_and, n_bmul, total shifted-value terms) per unit, marginal (boundary-free).
const COST_VISION4_PERM: (u64, u64, u64) = (320, 576, 10328);
const COST_VISION6_PERM: (u64, u64, u64) = (480, 960, 13764);
/// One SHA-256 compression block, two-lane-packed (`sha256_compress_2x_seq` / 2) — the
/// cheapest known in-circuit form, applicable to both Merkle nodes and sequential
/// challenger chains.
const COST_SHA256_BLOCK_2X: (u64, u64, u64) = (495, 0, 5997);
/// One SHA-256 compression block, plain single-lane gadget (upper bound).
const COST_SHA256_BLOCK_1X: (u64, u64, u64) = (907, 0, 11629);

// ---------------------------------------------------------------------------
// Counting SHA-256 digest wrapper
// ---------------------------------------------------------------------------

/// Global counters for one instrumentation role.
struct RoleCounters {
	/// Total bytes fed through `update`.
	bytes: AtomicU64,
	/// Number of digest finalizations (= complete SHA-256 messages hashed).
	finalizes: AtomicU64,
	/// Exact SHA-256 compression blocks consumed: per message, ceil((len + 9) / 64).
	blocks: AtomicU64,
	/// Histogram of finalized message lengths (bytes -> count).
	msg_lens: Mutex<BTreeMap<u64, u64>>,
}

impl RoleCounters {
	const fn new() -> Self {
		Self {
			bytes: AtomicU64::new(0),
			finalizes: AtomicU64::new(0),
			blocks: AtomicU64::new(0),
			msg_lens: Mutex::new(BTreeMap::new()),
		}
	}

	fn reset(&self) {
		self.bytes.store(0, Relaxed);
		self.finalizes.store(0, Relaxed);
		self.blocks.store(0, Relaxed);
		self.msg_lens.lock().unwrap().clear();
	}

	fn record_finalize(&self, len: u64) {
		self.finalizes.fetch_add(1, Relaxed);
		self.blocks.fetch_add((len + 9).div_ceil(64), Relaxed);
		*self.msg_lens.lock().unwrap().entry(len).or_insert(0) += 1;
	}
}

static CHAL_COUNTERS: RoleCounters = RoleCounters::new();
static MERKLE_LEAF_COUNTERS: RoleCounters = RoleCounters::new();
static MERKLE_COMPRESS_CALLS: AtomicU64 = AtomicU64::new(0);

trait Role: Clone + Default + Send + Sync + 'static {
	fn counters() -> &'static RoleCounters;
}

#[derive(Clone, Default)]
struct ChalRole;
impl Role for ChalRole {
	fn counters() -> &'static RoleCounters {
		&CHAL_COUNTERS
	}
}

#[derive(Clone, Default)]
struct MerkleLeafRole;
impl Role for MerkleLeafRole {
	fn counters() -> &'static RoleCounters {
		&MERKLE_LEAF_COUNTERS
	}
}

/// Byte-transparent SHA-256 wrapper that records message lengths and exact
/// compression-block counts per finalization into its role's global counters.
#[derive(Clone, Default)]
struct CountingSha256<R: Role> {
	inner: Sha256,
	len: u64,
	_role: PhantomData<R>,
}

impl<R: Role> HashMarker for CountingSha256<R> {}

impl<R: Role> Update for CountingSha256<R> {
	fn update(&mut self, data: &[u8]) {
		self.len += data.len() as u64;
		R::counters().bytes.fetch_add(data.len() as u64, Relaxed);
		Update::update(&mut self.inner, data);
	}
}

impl<R: Role> OutputSizeUser for CountingSha256<R> {
	type OutputSize = <Sha256 as OutputSizeUser>::OutputSize;
}

impl<R: Role> BlockSizeUser for CountingSha256<R> {
	type BlockSize = <Sha256 as BlockSizeUser>::BlockSize;
}

impl<R: Role> FixedOutput for CountingSha256<R> {
	fn finalize_into(self, out: &mut Output<Self>) {
		R::counters().record_finalize(self.len);
		FixedOutput::finalize_into(self.inner, out);
	}
}

impl<R: Role> FixedOutputReset for CountingSha256<R> {
	fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
		R::counters().record_finalize(self.len);
		self.len = 0;
		FixedOutputReset::finalize_into_reset(&mut self.inner, out);
	}
}

impl<R: Role> Reset for CountingSha256<R> {
	fn reset(&mut self) {
		self.len = 0;
		Reset::reset(&mut self.inner);
	}
}

// ---------------------------------------------------------------------------
// Counting Merkle hash suite (identical bytes to Sha256HashSuite)
// ---------------------------------------------------------------------------

/// Byte-transparent 2-to-1 compression wrapper counting invocations.
#[derive(Clone, Default)]
struct CountingCompression(Sha256Compression);

impl CompressionFunction<Output<Sha256>, 2> for CountingCompression {
	fn compress(&self, input: [Output<Sha256>; 2]) -> Output<Sha256> {
		MERKLE_COMPRESS_CALLS.fetch_add(1, Relaxed);
		self.0.compress(input)
	}
}

/// Hash suite producing bytes identical to [`Sha256HashSuite`] while counting all
/// verifier-side leaf hashes and inner-node compressions.
#[derive(Clone)]
struct CountingSuite;

impl HashSuite for CountingSuite {
	type LeafHash = CountingSha256<MerkleLeafRole>;
	type Compression = CountingCompression;
	type ParLeafHash = ParallelDigestAdapter<CountingSha256<MerkleLeafRole>>;
	type ParCompression = ParallelCompressionAdaptor<CountingCompression>;
}

// ---------------------------------------------------------------------------
// Counting challenger: HasherChallenger<Sha256> + protocol-level phase log
// ---------------------------------------------------------------------------
//
// A verbatim structural copy of `binius_transcript::fiat_shamir::HasherChallenger`
// specialized to a counting SHA-256, with the protocol-facing byte streams logged in
// `advance`/`advance_mut`. Behavior is pinned byte-identical to the original by
// `counting_challenger_matches_std` below and by every instrumented verification passing.

type CSha = CountingSha256<ChalRole>;

/// (kind, bytes): kind 0 = observe (prover message absorbed), 1 = sample (challenge bytes).
static PHASE_LOG: Mutex<Vec<(u8, u64)>> = Mutex::new(Vec::new());

fn log_phase_bytes(kind: u8, n: u64) {
	if n == 0 {
		return;
	}
	let mut log = PHASE_LOG.lock().unwrap();
	match log.last_mut() {
		Some((k, b)) if *k == kind => *b += n,
		_ => log.push((kind, n)),
	}
}

#[derive(Clone, Default)]
struct CountingSampler {
	index: usize,
	buffer: Output<CSha>,
	hasher: CSha,
}

#[derive(Clone, Default)]
struct CountingObserver {
	index: usize,
	buffer: Block<CSha>,
	hasher: CSha,
}

#[derive(Clone)]
enum CountingChallenger {
	Observer(CountingObserver),
	Sampler(CountingSampler),
}

impl Default for CountingChallenger {
	fn default() -> Self {
		let initial_digest = <CSha as Digest>::digest([]);
		let mut hasher = <CSha as Digest>::new();
		Digest::update(&mut hasher, &initial_digest);
		Self::Sampler(CountingSampler {
			hasher,
			index: 0,
			buffer: initial_digest,
		})
	}
}

impl Challenger for CountingChallenger {
	fn observer(&mut self) -> &mut impl BufMut {
		match self {
			Self::Observer(observer) => observer,
			Self::Sampler(sampler) => {
				*self = Self::Observer(mem::take(sampler).into_observer());
				match self {
					Self::Observer(observer) => observer,
					_ => unreachable!(),
				}
			}
		}
	}

	fn sampler(&mut self) -> &mut impl Buf {
		match self {
			Self::Sampler(sampler) => sampler,
			Self::Observer(observer) => {
				*self = Self::Sampler(mem::take(observer).into_sampler());
				match self {
					Self::Sampler(sampler) => sampler,
					_ => unreachable!(),
				}
			}
		}
	}
}

impl CountingSampler {
	fn into_observer(mut self) -> CountingObserver {
		Digest::update(&mut self.hasher, (self.index as u64).to_le_bytes());
		CountingObserver {
			hasher: self.hasher,
			index: 0,
			buffer: Block::<CSha>::default(),
		}
	}

	fn fill_buffer(&mut self) {
		let digest = self.hasher.finalize_reset();
		// Feed forward to the empty state.
		Digest::update(&mut self.hasher, &digest);
		self.buffer = digest;
		self.index = 0;
	}
}

impl CountingObserver {
	fn into_sampler(mut self) -> CountingSampler {
		self.flush();
		CountingSampler {
			hasher: self.hasher,
			index: <CSha as Digest>::output_size(),
			buffer: Output::<CSha>::default(),
		}
	}

	fn flush(&mut self) {
		let buf = self.buffer.clone();
		Update::update(&mut self.hasher, &buf[..self.index]);
		self.index = 0;
	}
}

impl Buf for CountingSampler {
	fn remaining(&self) -> usize {
		usize::MAX
	}

	fn chunk(&self) -> &[u8] {
		&self.buffer[self.index..]
	}

	fn advance(&mut self, mut cnt: usize) {
		log_phase_bytes(1, cnt as u64);
		if self.index == <CSha as Digest>::output_size() {
			self.fill_buffer();
		}
		while cnt > 0 {
			let remaining = cnt.min(<CSha as Digest>::output_size() - self.index);
			if remaining == 0 {
				self.fill_buffer();
				continue;
			}
			cnt -= remaining;
			self.index += remaining;
		}
	}
}

unsafe impl BufMut for CountingObserver {
	fn remaining_mut(&self) -> usize {
		usize::MAX
	}

	unsafe fn advance_mut(&mut self, mut cnt: usize) {
		log_phase_bytes(0, cnt as u64);
		while cnt > 0 {
			let remaining = cnt.min(<CSha as BlockSizeUser>::block_size() - self.index);
			cnt -= remaining;
			self.index += remaining;
			if self.index == <CSha as BlockSizeUser>::block_size() {
				self.flush();
			}
		}
	}

	fn chunk_mut(&mut self) -> &mut UninitSlice {
		let buffer = &mut self.buffer[self.index..];
		buffer.into()
	}
}

fn reset_all_counters() {
	CHAL_COUNTERS.reset();
	MERKLE_LEAF_COUNTERS.reset();
	MERKLE_COMPRESS_CALLS.store(0, Relaxed);
	PHASE_LOG.lock().unwrap().clear();
}

// ---------------------------------------------------------------------------
// Equivalence test (fast, always run)
// ---------------------------------------------------------------------------

/// Byte-for-byte equivalence of the counting challenger with the standard
/// `HasherChallenger<Sha256>` over a mixed observe/sample schedule.
#[test]
fn counting_challenger_matches_std() {
	let mut counting = CountingChallenger::default();
	let mut std_chal = HasherChallenger::<Sha256>::default();
	let mut rng = StdRng::seed_from_u64(0xC0FFEE);

	for round in 0..64 {
		if rng.random::<u32>() % 2 == 0 {
			let n = 1 + (rng.random::<u32>() as usize % 97);
			let data: Vec<u8> = (0..n).map(|_| rng.random()).collect();
			counting.observer().put_slice(&data);
			std_chal.observer().put_slice(&data);
		} else {
			let n = 1 + (rng.random::<u32>() as usize % 79);
			let mut a = vec![0u8; n];
			let mut b = vec![0u8; n];
			counting.sampler().copy_to_slice(&mut a);
			std_chal.sampler().copy_to_slice(&mut b);
			assert_eq!(a, b, "challenger divergence at round {round}");
		}
	}
}

// ---------------------------------------------------------------------------
// Shared circuit builders
// ---------------------------------------------------------------------------

/// A SHA-256 message-hash leaf circuit (in-tree `sha256` example shape).
struct ShaLeaf {
	circuit: binius_frontend::Circuit,
	message: Vec<Wire>,
	digest: [Wire; 8],
	len_bytes: usize,
}

fn build_sha_leaf(len_bytes: usize) -> ShaLeaf {
	let b = CircuitBuilder::new();
	let n_words = len_bytes.div_ceil(4);
	let message: Vec<Wire> = (0..n_words).map(|_| b.add_inout()).collect();
	let computed = sha256_fixed(&b, &message, len_bytes);
	let digest: [Wire; 8] = array::from_fn(|_| b.add_inout());
	for i in 0..8 {
		b.assert_eq(format!("digest[{i}]"), computed[i], digest[i]);
	}
	ShaLeaf {
		circuit: b.build(),
		message,
		digest,
		len_bytes,
	}
}

fn populate_sha_leaf(leaf: &ShaLeaf) -> binius_core::constraint_system::ValueVec {
	let mut rng = StdRng::seed_from_u64(7);
	let msg: Vec<u8> = (0..leaf.len_bytes).map(|_| rng.random()).collect();
	let mut w = leaf.circuit.new_witness_filler();
	for (wire, word) in leaf.message.iter().zip(pack_bytes_u32words(&msg, true)) {
		w[*wire] = word;
	}
	let dig = sha2::Sha256::digest(&msg);
	for (wire, word) in leaf.digest.iter().zip(pack_bytes_u32words(&dig, true)) {
		w[*wire] = word;
	}
	leaf.circuit.populate_wire_witness(&mut w).unwrap();
	w.into_value_vec()
}

// ---------------------------------------------------------------------------
// P3b — transcript node model
// ---------------------------------------------------------------------------

/// Vision-6 sponge permutations to absorb-and-finalize a message of `len` bytes
/// (rate 64 B; keccak-style padding always costs one permutation, matching the
/// restored `vision_6::digest::VisionHasherDigest`).
const fn vision6_digest_perms(len: u64) -> u64 {
	len / 64 + 1
}

fn print_cost(label: &str, n_units: u64, unit: (u64, u64, u64)) -> (u64, u64, u64) {
	let (and, bmul, terms) = (n_units * unit.0, n_units * unit.1, n_units * unit.2);
	println!("  {label:<44} {n_units:>7} units -> {and:>9} AND {bmul:>9} BMUL {terms:>11} terms");
	(and, bmul, terms)
}

#[test]
#[ignore = "probe measurement: run explicitly with --release --ignored --nocapture"]
fn probe_transcript_node_model() {
	for len_bytes in [1024usize, 4096, 16384, 65536] {
		let leaf = build_sha_leaf(len_bytes);
		let witness = populate_sha_leaf(&leaf);
		let cs: ConstraintSystem = leaf.circuit.constraint_system().clone();

		// Leaf-scale bookkeeping for the node/leaf fraction.
		let leaf_n_and = cs.and_constraints.len() as u64;
		let leaf_and_terms: u64 = cs
			.and_constraints
			.iter()
			.map(|c| (c.a.len() + c.b.len() + c.c.len()) as u64)
			.sum();

		let verifier = Verifier::<Sha256HashSuite>::setup(cs.clone(), LOG_INV_RATE).unwrap();
		let prover = Prover::<OptimalPackedB128, Sha256HashSuite>::setup(verifier.clone()).unwrap();

		let mut pt = ProverTranscript::new(StdChallenger::default());
		prover.prove(witness.clone(), &mut pt).unwrap();
		let proof = pt.finalize();

		// Sanity: the standard verifier accepts.
		let mut vt = VerifierTranscript::new(StdChallenger::default(), proof.clone());
		verifier.verify(witness.public(), &mut vt).unwrap();
		vt.finalize().unwrap();

		// Instrumented verification (byte-identical hashing, counted).
		let counting_verifier = Verifier::<CountingSuite>::setup(cs, LOG_INV_RATE).unwrap();
		reset_all_counters();
		let mut vt = VerifierTranscript::new(CountingChallenger::default(), proof.clone());
		counting_verifier
			.verify(witness.public(), &mut vt)
			.expect("counting verifier must accept: wrappers are byte-transparent");
		vt.finalize().unwrap();

		// -- Snapshot counters --------------------------------------------------
		let phases = PHASE_LOG.lock().unwrap().clone();
		let n_observe_phases = phases.iter().filter(|(k, _)| *k == 0).count() as u64;
		let n_sample_phases = phases.iter().filter(|(k, _)| *k == 1).count() as u64;
		let observe_bytes: u64 = phases.iter().filter(|(k, _)| *k == 0).map(|(_, b)| b).sum();
		let sample_bytes: u64 = phases.iter().filter(|(k, _)| *k == 1).map(|(_, b)| b).sum();

		// The challenger's Default consumes one constant compression (SHA-256 of the
		// empty string); it is a compile-time constant in any arithmetization, so it
		// is reported separately and excluded from the model.
		let chal_finalizes = CHAL_COUNTERS.finalizes.load(Relaxed) - 1;
		let chal_blocks = CHAL_COUNTERS.blocks.load(Relaxed) - 1;

		let leaf_hashes = MERKLE_LEAF_COUNTERS.finalizes.load(Relaxed);
		let leaf_blocks = MERKLE_LEAF_COUNTERS.blocks.load(Relaxed);
		let leaf_lens = MERKLE_LEAF_COUNTERS.msg_lens.lock().unwrap().clone();
		let compress_calls = MERKLE_COMPRESS_CALLS.load(Relaxed);

		println!();
		println!("==========================================================================");
		println!(
			"P3b probe @ leaf message {len_bytes} B | leaf circuit: {leaf_n_and} AND, {leaf_and_terms} terms | log_witness_words={} | proof {} B",
			counting_verifier.log_witness_words(),
			proof.len(),
		);
		println!("--- measured verifier transcript traffic (SHA-256 base layer) ---");
		println!(
			"  challenger: {n_observe_phases} observe phases ({observe_bytes} B absorbed), {n_sample_phases} sample phases ({sample_bytes} B sampled)"
		);
		println!(
			"  challenger SHA-256: {chal_finalizes} finalizations, {chal_blocks} compression blocks (constant initial block excluded)"
		);
		println!(
			"  merkle: {leaf_hashes} leaf hashes ({leaf_blocks} SHA blocks), {compress_calls} inner 2-to-1 compressions"
		);
		println!("  merkle leaf message lengths: {leaf_lens:?}");
		let total_sha_blocks = chal_blocks + leaf_blocks + compress_calls;
		println!("  TOTAL SHA-256 compressions per verify: {total_sha_blocks}");

		// The public statement is absorbed into the challenger. In THIS leaf circuit the
		// statement is the whole hashed message (inout wires) — large. At a recursion
		// node the child statement is a digest-sized claim, so model both: the raw
		// measurement, and a protocol-only variant with a 128-B child statement.
		let stmt_bytes = (1u64 << counting_verifier.log_public_words()) * 8;
		let first_observe = phases
			.iter()
			.find(|(k, _)| *k == 0)
			.map(|(_, b)| *b)
			.unwrap_or(0);
		assert!(
			first_observe >= stmt_bytes,
			"statement expected inside the first observe phase ({first_observe} < {stmt_bytes})"
		);
		const CHILD_STMT_BYTES: u64 = 128;
		let protocol_observe = observe_bytes - stmt_bytes;
		println!(
			"  public statement: {stmt_bytes} B of the absorb traffic ({} SHA blocks); protocol-only absorb: {protocol_observe} B",
			stmt_bytes / 64
		);

		// Per-phase byte log with the statement swapped for a 128-B recursion claim.
		let recursion_phases: Vec<(u8, u64)> = {
			let mut swapped = false;
			phases
				.iter()
				.map(|&(k, b)| {
					if k == 0 && !swapped {
						swapped = true;
						(k, b - stmt_bytes + CHILD_STMT_BYTES)
					} else {
						(k, b)
					}
				})
				.collect()
		};
		let recursion_sha_blocks =
			total_sha_blocks - stmt_bytes / 64 + CHILD_STMT_BYTES.div_ceil(64);

		// -- Node model ---------------------------------------------------------
		// SHA-256 arithmetization: every counted compression block costs one in-circuit
		// block (2x-packed rate, with the 1x rate as upper bound).
		println!("--- node model: in-circuit cost of ONE child-proof transcript ---");
		println!("  [SHA-256 transcript, 2x-packed blocks @ {:?}]", COST_SHA256_BLOCK_2X);
		let sha_total =
			print_cost("all SHA blocks (raw stmt)", total_sha_blocks, COST_SHA256_BLOCK_2X);
		let sha_rec = print_cost(
			"all SHA blocks (128-B child stmt)",
			recursion_sha_blocks,
			COST_SHA256_BLOCK_2X,
		);
		println!("  [SHA-256 transcript, single-lane upper bound @ {:?}]", COST_SHA256_BLOCK_1X);
		let _ = print_cost("all SHA blocks (raw stmt, 1x)", total_sha_blocks, COST_SHA256_BLOCK_1X);

		// Vision arithmetization of the SAME protocol traffic:
		// - challenger -> Vision-6 duplex at rate 64 B: per phase ceil(bytes/64) permutations (each
		//   phase boundary costs at least one permutation);
		// - merkle leaf hash -> Vision-6 digest of the same leaf bytes;
		// - merkle 2-to-1 -> one Vision-4 permutation (MMO), per the restored VisionHashSuite.
		let v6_phase_perms =
			|ph: &[(u8, u64)]| -> u64 { ph.iter().map(|(_, b)| b.div_ceil(64).max(1)).sum() };
		let chal_v6 = v6_phase_perms(&phases);
		let chal_v6_rec = v6_phase_perms(&recursion_phases);
		let leaf_v6: u64 = leaf_lens
			.iter()
			.map(|(len, count)| vision6_digest_perms(*len) * count)
			.sum();
		println!("  [Vision transcript @ v4 {COST_VISION4_PERM:?}, v6 {COST_VISION6_PERM:?}]");
		let v6_chal =
			print_cost("challenger Vision-6 perms (raw stmt)", chal_v6, COST_VISION6_PERM);
		let v6_chal_rec =
			print_cost("challenger Vision-6 perms (128-B stmt)", chal_v6_rec, COST_VISION6_PERM);
		let v6_leaf = print_cost("merkle-leaf Vision-6 perms", leaf_v6, COST_VISION6_PERM);
		let v4_nodes =
			print_cost("merkle 2-to-1 Vision-4 perms", compress_calls, COST_VISION4_PERM);
		let vision_total = (
			v6_chal.0 + v6_leaf.0 + v4_nodes.0,
			v6_chal.1 + v6_leaf.1 + v4_nodes.1,
			v6_chal.2 + v6_leaf.2 + v4_nodes.2,
		);
		let vision_rec = (
			v6_chal_rec.0 + v6_leaf.0 + v4_nodes.0,
			v6_chal_rec.1 + v6_leaf.1 + v4_nodes.1,
			v6_chal_rec.2 + v6_leaf.2 + v4_nodes.2,
		);
		println!(
			"  RAW STMT   — VISION: {} AND + {} BMUL = {} terms | SHA-2x: {} AND = {} terms | Vision/SHA {:.3}",
			vision_total.0,
			vision_total.1,
			vision_total.2,
			sha_total.0,
			sha_total.2,
			vision_total.2 as f64 / sha_total.2 as f64,
		);
		println!(
			"  128-B STMT — VISION: {} AND + {} BMUL = {} terms | SHA-2x: {} AND = {} terms | Vision/SHA {:.3}",
			vision_rec.0,
			vision_rec.1,
			vision_rec.2,
			sha_rec.0,
			sha_rec.2,
			vision_rec.2 as f64 / sha_rec.2 as f64,
		);
		println!(
			"  node-transcript terms as fraction of THIS leaf's terms (x2 children, 128-B stmt): SHA {:.3}, Vision {:.3}",
			2.0 * sha_rec.2 as f64 / leaf_and_terms as f64,
			2.0 * vision_rec.2 as f64 / leaf_and_terms as f64,
		);
	}
}

// ---------------------------------------------------------------------------
// P3a.2 — prover/verifier wall-time per constraint class
// ---------------------------------------------------------------------------

struct TimedRun {
	label: String,
	n_and: usize,
	n_bmul: usize,
	value_vec_len: usize,
	populate: f64,
	prove_median: f64,
	verify_median: f64,
}

fn median(mut xs: Vec<f64>) -> f64 {
	xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
	xs[xs.len() / 2]
}

/// A populate closure filling the input wires the build closure created.
type Populator = Box<dyn for<'a> Fn(&mut binius_frontend::WitnessFiller<'a>)>;

fn time_circuit(label: &str, build: impl FnOnce(&CircuitBuilder) -> Populator) -> TimedRun {
	let b = CircuitBuilder::new();
	let populator = build(&b);
	let circuit = b.build();
	let cs = circuit.constraint_system().clone();
	let n_and = cs.and_constraints.len();
	let n_bmul = cs.bmul_constraints.len();

	let t0 = Instant::now();
	let mut w = circuit.new_witness_filler();
	populator(&mut w);
	circuit.populate_wire_witness(&mut w).unwrap();
	let witness = w.into_value_vec();
	let populate_time = t0.elapsed().as_secs_f64();
	let value_vec_len = witness.combined_witness().len();

	let verifier = Verifier::<Sha256HashSuite>::setup(cs, LOG_INV_RATE).unwrap();
	let prover = Prover::<OptimalPackedB128, Sha256HashSuite>::setup(verifier.clone()).unwrap();

	let mut prove_times = Vec::new();
	let mut proof = Vec::new();
	for _ in 0..3 {
		let mut pt = ProverTranscript::new(StdChallenger::default());
		let t0 = Instant::now();
		prover.prove(witness.clone(), &mut pt).unwrap();
		prove_times.push(t0.elapsed().as_secs_f64());
		proof = pt.finalize();
	}

	let mut verify_times = Vec::new();
	for _ in 0..3 {
		let mut vt = VerifierTranscript::new(StdChallenger::default(), proof.clone());
		let t0 = Instant::now();
		verifier.verify(witness.public(), &mut vt).unwrap();
		verify_times.push(t0.elapsed().as_secs_f64());
		vt.finalize().unwrap();
	}

	let run = TimedRun {
		label: label.to_string(),
		n_and,
		n_bmul,
		value_vec_len,
		populate: populate_time,
		prove_median: median(prove_times),
		verify_median: median(verify_times),
	};
	println!(
		"  [done] {}: {} AND, {} BMUL, vv {} | populate {:.3}s prove {:.3}s verify {:.4}s",
		run.label,
		run.n_and,
		run.n_bmul,
		run.value_vec_len,
		run.populate,
		run.prove_median,
		run.verify_median
	);
	run
}

fn sha_leaf_builder(len_bytes: usize) -> impl FnOnce(&CircuitBuilder) -> Populator {
	move |b: &CircuitBuilder| {
		let n_words = len_bytes.div_ceil(4);
		let message: Vec<Wire> = (0..n_words).map(|_| b.add_inout()).collect();
		let computed = sha256_fixed(b, &message, len_bytes);
		let digest: [Wire; 8] = array::from_fn(|_| b.add_inout());
		for i in 0..8 {
			b.assert_eq(format!("digest[{i}]"), computed[i], digest[i]);
		}
		Box::new(move |w| {
			let mut rng = StdRng::seed_from_u64(7);
			let msg: Vec<u8> = (0..len_bytes).map(|_| rng.random()).collect();
			for (wire, word) in message.iter().zip(pack_bytes_u32words(&msg, true)) {
				w[*wire] = word;
			}
			let dig = sha2::Sha256::digest(&msg);
			for (wire, word) in digest.iter().zip(pack_bytes_u32words(&dig, true)) {
				w[*wire] = word;
			}
		})
	}
}

fn vision_chain_builder<const M: usize>(
	k: usize,
	perm: impl Fn(&CircuitBuilder, [GhashWire; M]) -> [GhashWire; M] + 'static,
) -> impl FnOnce(&CircuitBuilder) -> Populator {
	move |b: &CircuitBuilder| {
		let input: [GhashWire; M] = array::from_fn(|_| GhashWire::witness(b));
		let mut state = input;
		for _ in 0..k {
			state = perm(b, state);
		}
		for o in &state {
			o.force_commit(b);
		}
		Box::new(move |w| {
			let mut rng = StdRng::seed_from_u64(11);
			for wire in &input {
				wire.populate(w, Ghash::random(&mut rng));
			}
		})
	}
}

#[test]
#[ignore = "probe measurement: run explicitly with --release --ignored --nocapture"]
fn probe_prove_verify_timing() {
	println!();
	println!("P3a.2 — prove/verify wall time (single-threaded; medians of 3)");
	let runs = vec![
		time_circuit("sha256 8 KiB (129 blocks)", sha_leaf_builder(8192)),
		time_circuit("sha256 16 KiB (257 blocks)", sha_leaf_builder(16384)),
		time_circuit("vision4 x128", vision_chain_builder::<4>(128, vision4_permutation)),
		time_circuit("vision4 x256", vision_chain_builder::<4>(256, vision4_permutation)),
		time_circuit("vision6 x128", vision_chain_builder::<6>(128, vision6_permutation)),
		time_circuit("vision6 x256", vision_chain_builder::<6>(256, vision6_permutation)),
	];

	println!();
	println!(
		"| circuit                    | n_and  | n_bmul | value_vec | populate s | prove s | verify s |"
	);
	println!(
		"|----------------------------|--------|--------|-----------|------------|---------|----------|"
	);
	for r in &runs {
		println!(
			"| {:<26} | {:>6} | {:>6} | {:>9} | {:>10.3} | {:>7.3} | {:>8.4} |",
			r.label,
			r.n_and,
			r.n_bmul,
			r.value_vec_len,
			r.populate,
			r.prove_median,
			r.verify_median
		);
	}

	// Marginal per-unit prover cost from the size differences.
	let marg = |big: &TimedRun, small: &TimedRun, units: f64| {
		((big.prove_median - small.prove_median) / units) * 1e6
	};
	println!();
	println!("marginal prove cost:");
	println!("  sha256: {:.1} us/block (128 extra blocks)", marg(&runs[1], &runs[0], 128.0));
	println!("  vision4: {:.1} us/perm (128 extra perms)", marg(&runs[3], &runs[2], 128.0));
	println!("  vision6: {:.1} us/perm (128 extra perms)", marg(&runs[5], &runs[4], 128.0));
}

// ---------------------------------------------------------------------------
// P4 — native throughput
// ---------------------------------------------------------------------------

fn bench_loop(label: &str, mut iter_once: impl FnMut() -> u64) -> f64 {
	// Warm up.
	for _ in 0..3 {
		iter_once();
	}
	let t0 = Instant::now();
	let mut units = 0u64;
	let mut iters = 0u64;
	while t0.elapsed().as_secs_f64() < 1.0 {
		units += iter_once();
		iters += 1;
	}
	let secs = t0.elapsed().as_secs_f64();
	let per_unit_ns = secs * 1e9 / units as f64;
	println!(
		"  {label:<44} {units:>12} units in {secs:>6.3} s  ({per_unit_ns:>9.1} ns/unit, {iters} iters)"
	);
	per_unit_ns
}

#[test]
#[ignore = "probe measurement: run explicitly with --release --ignored --nocapture"]
fn probe_native_throughput() {
	use std::mem::MaybeUninit;

	let mut rng = StdRng::seed_from_u64(3);

	println!();
	println!("P4 — native throughput (single-threaded; rayon shim = serial in this build)");

	// (a) Merkle 2-to-1 compression, scalar.
	let mk_digest = |rng: &mut StdRng| {
		let mut d = Output::<Sha256>::default();
		for byte in d.iter_mut() {
			*byte = rng.random();
		}
		d
	};
	let pairs: Vec<[Output<Sha256>; 2]> = (0..1024)
		.map(|_| [mk_digest(&mut rng), mk_digest(&mut rng)])
		.collect();

	let sha_c = Sha256Compression::default();
	let sha_scalar = bench_loop("sha256 2-to-1 scalar (hw compress256)", || {
		for p in &pairs {
			std::hint::black_box(sha_c.compress(std::hint::black_box(p.clone())));
		}
		pairs.len() as u64
	});

	let vis_c = VisionCompression;
	let vis_scalar = bench_loop("vision-4 2-to-1 scalar", || {
		for p in &pairs {
			std::hint::black_box(vis_c.compress(std::hint::black_box(p.clone())));
		}
		pairs.len() as u64
	});

	// (a') batched paths (what the Merkle builder actually calls).
	let flat: Vec<Output<Sha256>> = pairs.iter().flat_map(|p| p.iter().cloned()).collect();
	let n_nodes = flat.len() / 2;

	let sha_par = ParallelSha256Compression::default();
	let sha_batched = bench_loop("sha256 2-to-1 batched x4 kernel", || {
		let mut out: Vec<MaybeUninit<Output<Sha256>>> =
			(0..n_nodes).map(|_| MaybeUninit::uninit()).collect();
		sha_par.parallel_compress(std::hint::black_box(&flat), &mut out);
		std::hint::black_box(&out);
		n_nodes as u64
	});

	let vis_par = VisionParallelCompression::default();
	let vis_batched = bench_loop("vision-4 2-to-1 batched (Montgomery x128)", || {
		let mut out: Vec<MaybeUninit<Output<Sha256>>> =
			(0..n_nodes).map(|_| MaybeUninit::uninit()).collect();
		vis_par.parallel_compress(std::hint::black_box(&flat), &mut out);
		std::hint::black_box(&out);
		n_nodes as u64
	});

	// (b) bulk hashing MB/s.
	let bulk: Vec<u8> = (0..1 << 20).map(|_| rng.random()).collect();
	let sha_bulk_ns = bench_loop("sha256 bulk hash 1 MiB", || {
		std::hint::black_box(sha2::Sha256::digest(std::hint::black_box(&bulk)));
		1
	});
	let vis_bulk_ns = bench_loop("vision-6 bulk hash 1 MiB (serial)", || {
		use binius_hash::vision_6::digest::VisionHasherDigest as V6;
		let mut h = <V6 as Digest>::new();
		Digest::update(&mut h, std::hint::black_box(&bulk));
		std::hint::black_box(h.finalize());
		1
	});

	// (b') batched Vision-6 leaf hashing (Montgomery batch inversion across 128 lanes) —
	// the actual prover-side leaf path of the restored VisionHashSuite. 128 leaves of
	// 256 B each per call = 32 KiB per call.
	{
		use std::mem::MaybeUninit;

		use binius_hash::{MultiDigest, vision_6::parallel_digest::VisionHasherMultiDigest};
		const N: usize = 128;
		const LEAF: usize = 256;
		let data: Vec<u8> = (0..N * LEAF).map(|_| rng.random()).collect();
		let refs: [&[u8]; N] = array::from_fn(|i| &data[i * LEAF..(i + 1) * LEAF]);
		let v6_batched_ns = bench_loop("vision-6 batched leaf hash (128x256 B)", || {
			let mut out = [MaybeUninit::uninit(); N];
			VisionHasherMultiDigest::<N, { N * 6 }>::digest(std::hint::black_box(refs), &mut out);
			std::hint::black_box(&out);
			(N * LEAF) as u64
		});
		// Per-byte comparison against hardware SHA-256 bulk rate.
		let sha_ns_per_byte = sha_bulk_ns / (1 << 20) as f64;
		println!(
			"  vision-6 batched: {:.1} MB/s ({:.1}x slower than sha256 bulk)",
			1e9 / v6_batched_ns / 1e6,
			v6_batched_ns / sha_ns_per_byte,
		);
	}

	println!();
	println!("--- summary ---");
	println!("  2-to-1 scalar slowdown (vision/sha):  {:.2}x", vis_scalar / sha_scalar);
	println!("  2-to-1 batched slowdown (vision/sha): {:.2}x", vis_batched / sha_batched);
	println!(
		"  bulk 1 MiB: sha256 {:.1} MB/s, vision-6 {:.1} MB/s, slowdown {:.2}x",
		(1 << 20) as f64 / (sha_bulk_ns / 1e9) / 1e6,
		(1 << 20) as f64 / (vis_bulk_ns / 1e9) / 1e6,
		vis_bulk_ns / sha_bulk_ns
	);
}
