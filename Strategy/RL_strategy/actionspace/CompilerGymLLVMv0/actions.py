from enum import Enum

class Actions_LLVM_10_0_0(Enum):
    AddDiscriminators = "--add-discriminators"
    Adce = "--adce"
    AggressiveInstcombine = "--aggressive-instcombine"
    AlignmentFromAssumptions = "--alignment-from-assumptions"
    AlwaysInline = "--always-inline"
    Argpromotion = "--argpromotion"
    Attributor = "--attributor"
    Barrier = "--barrier"
    Bdce = "--bdce"
    BreakCritEdges = "--break-crit-edges"
    Simplifycfg = "--simplifycfg"
    CallsiteSplitting = "--callsite-splitting"
    CalledValuePropagation = "--called-value-propagation"
    CanonicalizeAliases = "--canonicalize-aliases"
    Consthoist = "--consthoist"
    Constmerge = "--constmerge"
    Constprop = "--constprop"
    CoroCleanup = "--coro-cleanup"
    CoroEarly = "--coro-early"
    CoroElide = "--coro-elide"
    CoroSplit = "--coro-split"
    CorrelatedPropagation = "--correlated-propagation"
    CrossDsoCfi = "--cross-dso-cfi"
    Deadargelim = "--deadargelim"
    Dce = "--dce"
    Die = "--die"
    Dse = "--dse"
    Reg2mem = "--reg2mem"
    DivRemPairs = "--div-rem-pairs"
    EarlyCseMemssa = "--early-cse-memssa"
    EarlyCse = "--early-cse"
    ElimAvailExtern = "--elim-avail-extern"
    EeInstrument = "--ee-instrument"
    Flattencfg = "--flattencfg"
    Float2int = "--float2int"
    Forceattrs = "--forceattrs"
    Inline = "--inline"
    InsertGcovProfiling = "--insert-gcov-profiling"
    GvnHoist = "--gvn-hoist"
    Gvn = "--gvn"
    Globaldce = "--globaldce"
    Globalopt = "--globalopt"
    Globalsplit = "--globalsplit"
    Hotcoldsplit = "--hotcoldsplit"
    Ipconstprop = "--ipconstprop"
    Ipsccp = "--ipsccp"
    Indvars = "--indvars"
    Irce = "--irce"
    InferAddressSpaces = "--infer-address-spaces"
    Inferattrs = "--inferattrs"
    InjectTliMappings = "--inject-tli-mappings"
    Instsimplify = "--instsimplify"
    Instcombine = "--instcombine"
    Instnamer = "--instnamer"
    JumpThreading = "--jump-threading"
    Lcssa = "--lcssa"
    Licm = "--licm"
    LibcallsShrinkwrap = "--libcalls-shrinkwrap"
    LoadStoreVectorizer = "--load-store-vectorizer"
    LoopDataPrefetch = "--loop-data-prefetch"
    LoopDeletion = "--loop-deletion"
    LoopDistribute = "--loop-distribute"
    LoopFusion = "--loop-fusion"
    LoopIdiom = "--loop-idiom"
    LoopInstsimplify = "--loop-instsimplify"
    LoopInterchange = "--loop-interchange"
    LoopLoadElim = "--loop-load-elim"
    LoopPredication = "--loop-predication"
    LoopReroll = "--loop-reroll"
    LoopRotate = "--loop-rotate"
    LoopSimplifycfg = "--loop-simplifycfg"
    LoopSimplify = "--loop-simplify"
    LoopSink = "--loop-sink"
    LoopUnrollAndJam = "--loop-unroll-and-jam"
    LoopUnroll = "--loop-unroll"
    LoopUnswitch = "--loop-unswitch"
    LoopVectorize = "--loop-vectorize"
    LoopVersioningLicm = "--loop-versioning-licm"
    LoopVersioning = "--loop-versioning"
    Loweratomic = "--loweratomic"
    LowerConstantIntrinsics = "--lower-constant-intrinsics"
    LowerExpect = "--lower-expect"
    LowerGuardIntrinsic = "--lower-guard-intrinsic"
    Lowerinvoke = "--lowerinvoke"
    LowerMatrixIntrinsics = "--lower-matrix-intrinsics"
    Lowerswitch = "--lowerswitch"
    LowerWidenableCondition = "--lower-widenable-condition"
    Memcpyopt = "--memcpyopt"
    Mergefunc = "--mergefunc"
    Mergeicmps = "--mergeicmps"
    MldstMotion = "--mldst-motion"
    Sancov = "--sancov"
    NameAnonGlobals = "--name-anon-globals"
    NaryReassociate = "--nary-reassociate"
    Newgvn = "--newgvn"
    PgoMemopOpt = "--pgo-memop-opt"
    PartialInliner = "--partial-inliner"
    PartiallyInlineLibcalls = "--partially-inline-libcalls"
    PostInlineEeInstrument = "--post-inline-ee-instrument"
    Functionattrs = "--functionattrs"
    Mem2reg = "--mem2reg"
    PruneEh = "--prune-eh"
    Reassociate = "--reassociate"
    RedundantDbgInstElim = "--redundant-dbg-inst-elim"
    RpoFunctionattrs = "--rpo-functionattrs"
    RewriteStatepointsForGc = "--rewrite-statepoints-for-gc"
    Sccp = "--sccp"
    SlpVectorizer = "--slp-vectorizer"
    Sroa = "--sroa"
    Scalarizer = "--scalarizer"
    SeparateConstOffsetFromGep = "--separate-const-offset-from-gep"
    SimpleLoopUnswitch = "--simple-loop-unswitch"
    Sink = "--sink"
    SpeculativeExecution = "--speculative-execution"
    Slsr = "--slsr"
    StripDeadPrototypes = "--strip-dead-prototypes"
    StripDebugDeclare = "--strip-debug-declare"
    StripNondebug = "--strip-nondebug"
    Strip = "--strip"
    Tailcallelim = "--tailcallelim"
    Mergereturn = "--mergereturn"
