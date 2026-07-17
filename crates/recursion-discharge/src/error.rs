// Copyright 2026 The Binius Developers

//! Typed errors for the discharge verifiers.
//!
//! Every named variant corresponds to one verifier-side check, so the enum doubles as a
//! checklist of everything the verifiers enforce: the spec's P0.x structural
//! preconditions, the per-phase transcript checks, and the pinned-commitment bindings.
//! Sub-protocol rejections (sumcheck, fracaddcheck, BaseFold/FRI, channel, transcript)
//! are wrapped with the phase they occurred in.

/// Errors returned by [`crate::discharge::discharge_verify`],
/// [`crate::step2::discharge_verify_step2`], and
/// [`crate::merged::verify_merged_openings`].
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
	/// A structural precondition on the statement, VKM, or constraint system failed
	/// (the spec's P0.1–P0.4 admission checks, claim parsing included).
	#[error("{0}")]
	Precondition(String),
	/// Phase A: the received finish evaluations do not recombine to the batched
	/// sumcheck's reduced evaluation.
	#[error("phase A recombination failed (Horner check): claimed sums are inconsistent")]
	PhaseARecombination,
	/// Phase A: a claim's `g` evaluation does not equal `sum_o gamma_o * d_o`
	/// (the spec's (G) identity).
	#[error("phase A (G) recombination failed for claim {claim}")]
	PhaseGRecombination { claim: usize },
	/// Phase B: the final `m * W_eq(sigma) == e_B` check failed.
	#[error("phase B eq_ind final check failed")]
	PhaseBFinalCheck,
	/// Phase C: the sampled `tau` landed in the committed-value range (a possible pole);
	/// the honest prover aborts and the batch is re-proven.
	#[error("tau hit the committed-value range (honest abort)")]
	TauInPoleRange,
	/// Phase C: the received root denominator is zero (some denominator entry is zero).
	#[error("phase C root denominator is zero")]
	PhaseCRootDenominatorZero,
	/// Phase C: the fracaddcheck leaf point has the wrong arity for the union domain.
	#[error("phase C leaf point arity mismatch")]
	PhaseCLeafArity,
	/// Phase C: the numerator leaf claim does not match its transparent decomposition.
	#[error("phase C numerator transparent mismatch at pi")]
	PhaseCNumeratorTransparent,
	/// Phase C: the denominator leaf claim does not match its transparent decomposition.
	#[error("phase C denominator transparent mismatch at pi")]
	PhaseCDenominatorTransparent,
	/// STEP 1: the natively rebuilt `M_D` evaluation at `sigma` does not equal the
	/// prover's `m`.
	#[error("final native M_D check failed: prover's m does not match the CS-derived table")]
	NativeFinalCheck,
	/// The opened `M_VK` commitment does not equal the audited `vk_digest`
	/// (the P0.1 table-swap binding).
	#[error("P0.1: opened M_VK digest != audited vk_digest (table-swap)")]
	VkDigestPin,
	/// Merged opening: the received per-oracle evaluations do not recombine to the
	/// reduction sumcheck's reduced evaluation.
	#[error(
		"merged [M_VK, M_D] opening: reduction recombination failed \
		 (the point-evaluation claims are inconsistent)"
	)]
	ReductionRecombination,
	/// Merged opening: the final FRI oracle and the final sumcheck claim disagree.
	#[error("merged [M_VK, M_D] opening: FRI/MLE-check inconsistency")]
	FriMleInconsistency,
	/// A sub-protocol sumcheck verifier rejected.
	#[error("{phase}: {source}")]
	Sumcheck {
		phase: &'static str,
		source: binius_ip::sumcheck::Error,
	},
	/// The Phase C fracaddcheck verifier rejected.
	#[error("phase C fracaddcheck: {0}")]
	FracAdd(#[from] binius_ip::fracaddcheck::Error),
	/// A channel receive/assert failed (malformed or truncated proof data).
	#[error("{phase}: {source}")]
	Channel {
		phase: &'static str,
		source: binius_ip::channel::Error,
	},
	/// A raw transcript read failed (malformed or truncated proof data).
	#[error("{phase}: {source}")]
	Transcript {
		phase: &'static str,
		source: binius_transcript::Error,
	},
	/// The merged opening's Merkle channel rejected.
	#[error("merged [M_VK, M_D] opening: {0}")]
	MerkleChannel(#[from] binius_iop::merkle_channel::Error),
	/// The merged opening's combined FRI query/Merkle verification rejected.
	#[error("merged [M_VK, M_D] opening: FRI query/Merkle verification: {0}")]
	Basefold(#[from] binius_iop::basefold::Error),
}

/// `ensure!`-style helper returning [`VerifyError`] instead of `anyhow::Error`.
macro_rules! verify_ensure {
	($cond:expr, $err:expr) => {
		if !$cond {
			return Err($err);
		}
	};
}
pub(crate) use verify_ensure;
