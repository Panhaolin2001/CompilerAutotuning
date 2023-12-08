enum class LlvmAction {
  ADD_DISCRIMINATORS,
  ADCE,
  ALIGNMENT_FROM_ASSUMPTIONS,
  ALWAYS_INLINE,
  ANNOTATION2METADATA,
  ASSUME_BUILDER,
  ASSUME_SIMPLIFY,
  ATTRIBUTOR_CGSCC,
  ATTRIBUTOR,
  BARRIER,
  BDCE,
  BREAK_CRIT_EDGES,
  SIMPLIFYCFG,
  CALLSITE_SPLITTING,
  CALLED_VALUE_PROPAGATION,
  CANON_FREEZE,
  CONSTHOIST,
  CONSTMERGE,
  CORRELATED_PROPAGATION,
  CROSS_DSO_CFI,
  DFA_JUMP_THREADING,
  DEADARGELIM,
  DCE,
  DSE,
  DIV_REM_PAIRS,
  EARLY_CSE_MEMSSA,
  EARLY_CSE,
  ELIM_AVAIL_EXTERN,
  FIX_IRREDUCIBLE,
  FLATTENCFG,
  FLOAT2INT,
  FORCEATTRS,
  INLINE,
  GVN_HOIST,
  GVN,
  GLOBALDCE,
  GLOBALOPT,
  GLOBALSPLIT,
  GUARD_WIDENING,
  HOTCOLDSPLIT,
  IPSCCP,
  IROUTLINER,
  INDVARS,
  IRCE,
  INFER_ADDRESS_SPACES,
  INFERATTRS,
  INJECT_TLI_MAPPINGS,
  INSTSIMPLIFY,
  INSTCOMBINE,
  INSTNAMER,
  JUMP_THREADING,
  LCSSA,
  LICM,
  LIBCALLS_SHRINKWRAP,
  LOAD_STORE_VECTORIZER,
  LOOP_DATA_PREFETCH,
  LOOP_DELETION,
  LOOP_DISTRIBUTE,
  LOOP_EXTRACT,
  LOOP_FLATTEN,
  LOOP_FUSION,
  LOOP_GUARD_WIDENING,
  LOOP_IDIOM,
  LOOP_INSTSIMPLIFY,
  LOOP_INTERCHANGE,
  LOOP_LOAD_ELIM,
  LOOP_PREDICATION,
  LOOP_REROLL,
  LOOP_ROTATE,
  LOOP_SIMPLIFYCFG,
  LOOP_SIMPLIFY,
  LOOP_SINK,
  LOOP_REDUCE,
  LOOP_UNROLL_AND_JAM,
  LOOP_UNROLL,
  LOOP_VECTORIZE,
  LOOP_VERSIONING_LICM,
  LOOP_VERSIONING,
  LOWERATOMIC,
  LOWER_CONSTANT_INTRINSICS,
  LOWER_EXPECT,
  LOWER_GLOBAL_DTORS,
  LOWER_GUARD_INTRINSIC,
  LOWERINVOKE,
  LOWER_MATRIX_INTRINSICS_MINIMAL,
  LOWER_MATRIX_INTRINSICS,
  LOWERSWITCH,
  LOWER_WIDENABLE_CONDITION,
  MEMCPYOPT,
  MERGEFUNC,
  MERGEICMPS,
  MLDST_MOTION,
  NARY_REASSOCIATE,
  NEWGVN,
  OBJC_ARC_CONTRACT,
  PARTIAL_INLINER,
  PARTIALLY_INLINE_LIBCALLS,
  FUNCTION_ATTRS,
  MEM2REG,
  REASSOCIATE,
  REDUNDANT_DBG_INST_ELIM,
  REG2MEM,
  RPO_FUNCTION_ATTRS,
  REWRITE_STATEPOINTS_FOR_GC,
  SCCP,
  SLP_VECTORIZER,
  SROA,
  SCALARIZE_MASKED_MEM_INTRIN,
  SCALARIZER,
  SEPARATE_CONST_OFFSET_FROM_GEP,
  SIMPLE_LOOP_UNSWITCH,
  SINK,
  SPECULATIVE_EXECUTION,
  SLSR,
  STRIP_DEAD_PROTOTYPES,
  STRIP_DEBUG_DECLARE,
  STRIP_GC_RELOCATES,
  STRIP_NONDEBUG,
  STRIP_NONLINETABLE_DEBUGINFO,
  STRIP,
  STRUCTURIZECFG,
  TLSHOIST,
  TAILCALLELIM,
  MERGERETURN,
  UNIFY_LOOP_EXITS,
  VECTOR_COMBINE,
};
