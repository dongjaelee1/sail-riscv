// =======================================================================================
// This Sail RISC-V architecture model, comprising all files and
// directories except where otherwise noted is subject the BSD
// two-clause license in the LICENSE file.
//
// SPDX-License-Identifier: BSD-2-Clause
// =======================================================================================

// ****************************************************************
// Virtual memory address translation and memory protection,
// including PTWs (Page Table Walks) and TLBs (Translation Look-aside Buffers)
// Supported VS/S-stage modes: Sv32, Sv39, Sv48, Sv57
// Supported G-stage modes: Sv32x4, Sv39x4, Sv48x4, Sv57x4

// The code below implements the steps in the "Virtual Address
// Translation Process" (abbreviated here to VATP) section of the
// Privileged Architecture specification.  Code locations
// corresponding to these steps are marked in the comments.

// TLB NOTE:
// TLBs are not part of the RISC-V architecture specification.
// However, we model a simple TLB so that
// (1) we can meaningfully test SFENCE.VMA and HFENCE.GVMA/VVMA, which are
//     no-ops without TLBs;
// (2) we can greatly speed up simulation speed
//     (e.g., from 10s of minutes to few minutes for Linux boot)
// The TLB implementation is in a separate file: vmem_tlb.sail
// The code in this file is structured and commented so you can easily
// ignore TLB functionality at first reading.

struct PTW_Output('v : Int), is_sv_or_svx4_mode('v) = {
  ppn     : ppn_bits('v),
  pte     : pte_bits('v),
  pteAddr : physaddr,
  level   : level_range('v),
  pbmt    : page_based_mem_type,
  global  : bool,
}

private type PTW_Result('v : Int), is_sv_or_svx4_mode('v) = result((PTW_Output('v), ext_ptw), (PTW_Error, ext_ptw))

// The outcome of the atomic A/D-bit update of a leaf PTE (Step 7 of the
// VATP), when it is not a PTW_Error.  See update_and_write_pte().
union PTE_Update_Result('v : Int), is_sv_or_svx4_mode('v) = {
  // Some(pte): the PTE was written and now holds pte, the walked value
  // with its new A/D bits; None(): the walked value needs no new bits.
  PTE_Update_Done  : option(pte_bits('v)),
  // The PTE in memory differs from the value the walk used: the spec's
  // "return to step 2" -- the caller walks again.
  PTE_Update_Retry : unit,
}

// How many times a translation walks again after Step 7's comparison has
// failed before the model gives up.  The spec's loop is unbounded (a hart
// retries for as long as other agents keep changing the PTE); the model
// bounds it and stops the hart (internal_error) at the bound.  Any bound is
// a modelling choice; the retry count is the number of CONCURRENT changes
// to one PTE during one translation, so a small bound is never reached in
// practice.
let pte_update_max_retries : nat = 16

// Result of address translation
type TR_Result('paddr : Type, 'failure : Type) = result(('paddr, page_based_mem_type, ext_ptw), ('failure, ext_ptw))

// TODO: Remove this forward declaration if possible.
// G-Stage Translation
private val translate_g_stage : (
  gpabits,
  MemoryAccessType(mem_payload),
) -> TR_Result(gpabits, PTW_Error)

// ****************************************************************
// Page Table Walk (PTW)

// Write a Page Table Entry.
private function write_pte forall 'n, 'n in {4, 8} . (
  paddr    : physaddr,
  pte_size : int('n),
  pte      : bits('n * 8),
) -> MemoryOpResult(bool) =
  mem_write_value(paddr, pte_size, pte, Supervisor, Store(PageTableEntry), PBMT_PMA, false, false, false)

// Write a Page Table Entry as the write half of an atomic read-modify-write
// of that PTE.
private function write_pte_conditional forall 'n, 'n in {4, 8} . (
  paddr    : physaddr,
  pte_size : int('n),
  pte      : bits('n * 8),
) -> MemoryOpResult(bool) =
  mem_write_value(paddr, pte_size, pte, Supervisor, Store(PageTableEntry), PBMT_PMA, false, false, true)

// Sscptr: Main memory supports page table read
function clause currentlyEnabled(Ext_Ssccptr) = hartSupports(Ext_Ssccptr) & (currentlyEnabled(Ext_Sv32) | currentlyEnabled(Ext_Sv39))

// Read a Page Table Entry.
private function read_pte forall 'n, 'n in {4, 8} . (
  paddr    : physaddr,
  pte_size : int('n),
) -> MemoryOpResult(bits(8 * 'n)) =
  mem_read(Load(PageTableEntry), PBMT_PMA, Supervisor, paddr, pte_size, false, false, false)

// Read a Page Table Entry as the read half of an atomic read-modify-write
// of that PTE.
private function read_pte_exclusive forall 'n, 'n in {4, 8} . (
  paddr    : physaddr,
  pte_size : int('n),
) -> MemoryOpResult(bits(8 * 'n)) =
  mem_read(Load(PageTableEntry), PBMT_PMA, Supervisor, paddr, pte_size, false, false, true)

// Perform PTE address translation as part of VS-stage translation for implicit PTE accesses
private function translate_PTE_addr(
  pte_addr : gpabits,
  access   : MemoryAccessType(mem_payload),
  stage    : TranslationStage,
  ext_ptw  : ext_ptw
) -> TR_Result(physaddr, PTW_Error) = {
  if stage == VS_Stage then {
    match translate_g_stage(pte_addr, access) {
      Ok(pa, pbmt, ext_ptw) => Ok(Physaddr(pa), pbmt, ext_ptw),
      Err(f, ext_ptw) => Err(f, ext_ptw),
    }
  } else {
    Ok(Physaddr(pte_addr), PBMT_PMA, ext_ptw)
  }
}

termination_measure translate_PTE_addr(_, _, _, _) = 1000

// Check a leaf PTE and compute the translated PPN and the PBMT (Steps 3
// and 5-8 of the VATP; Step 4, following valid non-leaf PTEs, is handled
// in pt_walk()).  This is used by pt_walk() when it reaches a leaf PTE.
private function check_leaf_pte forall 'v, is_sv_or_svx4_mode('v) . (
  sv_width : int('v),
  vpn      : vpn_bits('v),
  access   : MemoryAccessType(mem_payload),
  priv     : Privilege,
  mxr      : bool,
  do_sum   : bool,
  pte      : pte_bits('v),
  pte_addr : physaddr,
  level    : level_range('v),
  stage    : TranslationStage,
  ext_ptw  : ext_ptw,
) -> result((ppn_bits('v), page_based_mem_type, ext_ptw), (PTW_Error, ext_ptw)) = {
  let pte_flags = Mk_PTE_Flags(pte[7 .. 0]);
  let pte_ext   = ext_bits_of_PTE(pte);

  // Step 3 of VATP.
  if pte_is_invalid(pte_flags, pte_ext, stage) then {
    ptw_fail_callback(stage, level, PTW_Invalid_PTE(), bits_of(pte_addr));
    return Err(PTW_Invalid_PTE(), ext_ptw);
  };

  // A non-leaf PTE is not a valid leaf: either this is a level 0 PTE that
  // contains a pointer, or (during an A/D-bit update) the PTE was
  // concurrently replaced by a pointer after the walk read a leaf here.
  if pte_is_non_leaf(pte_flags) then {
    ptw_fail_callback(stage, level, PTW_Invalid_PTE(), bits_of(pte_addr));
    return Err(PTW_Invalid_PTE(), ext_ptw);
  };

  // Leaf PTE (Step 5 of VATP).
  let ppn = PPN_of_PTE(pte); // 22 or 44.
  let 'vpn_i_size = get_vpn_level_size(sv_width);
  let low_bits = vpn_i_size * level;
  if level > 0 then {
    // Check for misaligned superpage.
    if ppn[low_bits - 1 .. 0] != zeros() then {
      ptw_fail_callback(stage, level, PTW_Misaligned(), bits_of(pte_addr));
      return Err(PTW_Misaligned(), ext_ptw);
    };
  };

  // Steps 6, 7 and 8 of VATP.
  match check_PTE_permission(access, priv, stage, mxr, do_sum, pte_flags, pte_ext, ext_ptw) {
    PTE_Check_Failure(ext_ptw, pte_failure) => {
      ptw_fail_callback(stage, level, ext_get_ptw_error(pte_failure), bits_of(pte_addr));
      Err(ext_get_ptw_error(pte_failure), ext_ptw)
    },
    PTE_Check_Success(ext_ptw) => {
      let ppn : ppn_bits('v) = if level > 0 then {
        // NAPOT PTEs are currently reserved for level > 0.
        if pte_ext[N] == 0b1 & pte_reserved_bits_must_be_zero
        then return Err(PTW_Invalid_PTE(), ext_ptw);

        // Compose final PA in superpage:
        // Superpage PPN @ lower VPNs @ page-offset
        ppn[length(ppn) - 1 .. low_bits] @ vpn[low_bits - 1 .. 0]
      } else if currentlyEnabled(Ext_Svnapot) & pte_ext[N] == 0b1 then {
        // Only 64K NAPOT PTEs are valid.
        let pte_napot_bits = 4;
        if ppn[pte_napot_bits - 1 .. 0] != 0b1000
        then return Err(PTW_Invalid_PTE(), ext_ptw);

        [ppn with (pte_napot_bits - 1) .. 0 = vpn[pte_napot_bits - 1 .. 0]]
      } else {
        ppn
      };

      let pbmt = if menvcfg[PBMTE] == 0b0 | (stage == VS_Stage & read_henvcfg()[PBMTE] == 0b0)
                 then PBMT_PMA
                 else page_based_mem_type(pte_ext[PBMT]);

      Ok((ppn, pbmt, ext_ptw))
    }
  }
}

// Update a PTE's A/D bits and write it to memory if needed.  This returns
// * Ok(PTE_Update_Done(Some(pte)), ext_ptw) if a write was needed and was
//   successful. pte is the new value.
// * Ok(PTE_Update_Done(None()), ext_ptw) if no write was needed.
// * Ok(PTE_Update_Retry(), ext_ptw) if a write was needed but the PTE in
//   memory no longer equals the value the walk used: the caller walks again.
// * Err(PTW_PTE_Needs_Update(), ext_ptw) if a write was needed but is forbidden by the current state.
// * Err(PTW_Implicit_Error(..), ext_ptw) if a write was needed but the G-stage
//   translation of the PTE's address failed (VS-stage only).
// * Err(PTW_No_Access(), ext_ptw) if a write was needed and attempted but
//   there was a physical memory access error.
//
// The privileged spec requires the A/D-bit update to be atomic:
//
//   "The PTE update must be atomic with respect to other accesses to the
//    PTE, and must atomically perform all tablewalk checks for that leaf
//    PTE as part of, and before, conditionally updating the PTE value."
//
// and Step 7 of the VATP specifies it as a compare-and-swap:
//
//   "Compare pte to the value of the PTE at address a+va.vpn[i]*PTESIZE.
//    If the values match, set pte.a to 1 and, if the original memory
//    access is a store, also set pte.d to 1.  If the comparison fails,
//    return to step 2."
//
// This is modelled here literally, as a single atomic compare-and-swap of
// the PTE: re-read the PTE with an exclusive read, compare it with the
// value the walk used (on which the tablewalk checks were performed), and
// if they are equal conditionally write back that value with its new A/D
// bits.  If they differ the PTE was changed since the walk read it, and
// the caller returns to Step 2 (walks again).
//
// pteAddr is the address of the PTE as seen by the page-table walk: a
// guest physical address for VS-stage translation (translated through the
// G-stage here, once, for both halves of the atomic update), and a
// supervisor physical address otherwise.
private function update_and_write_pte forall 'v, is_sv_or_svx4_mode('v) . (
  sv_width : int('v),
  pteAddr  : physaddr,
  pte      : pte_bits('v),
  access   : MemoryAccessType(mem_payload),
  stage    : TranslationStage,
  ext_ptw  : ext_ptw,
) -> result((PTE_Update_Result('v), ext_ptw), (PTW_Error, ext_ptw)) =
  match update_PTE_Bits(pte, access, stage) {
    None()        => Ok(PTE_Update_Done(None()), ext_ptw),
    Some(pte_new) => // The PTE needs new A/D bits.
      if hw_updates_pte(stage) then {
        let 'pte_width = if sv_width == 32 | sv_width == 34 then 4 else 8;
        // Translate PTE address if performing a VS-stage access.  The
        // atomic update is an implicit write of the PTE, so the G-stage
        // translation is performed for a store.
        let pte_paddr : physaddr = match translate_PTE_addr(bits_of(pteAddr), Store(PageTableEntry), stage, ext_ptw) {
          Ok(pa, _, _) => pa,
          Err(e, _)    => return Err(PTW_Implicit_Error(translationException(access, e), bits_of(pteAddr), Store(PageTableEntry)), ext_ptw),
        };
        // Step 7, atomically: re-read the PTE, compare it with the value
        // the walk used, and if they match write that value with its new
        // A/D bits.
        match read_pte_exclusive(pte_paddr, pte_width) {
          Err(_) => Err(PTW_No_Access(), ext_ptw),
          Ok(pte_fresh) =>
            if pte_fresh != pte then
              // The PTE was changed since the walk read it: "If the
              // comparison fails, return to step 2."
              Ok(PTE_Update_Retry(), ext_ptw)
            else
              match write_pte_conditional(pte_paddr, pte_width, pte_new) {
                Ok(true)  => Ok(PTE_Update_Done(Some(pte_new)), ext_ptw),
                // A failed conditional write would also be a "return to
                // step 2"; as for the AMO write in zaamo_insts.sail, it
                // cannot occur in the sequential emulator.
                Ok(false) => internal_error(__FILE__, __LINE__, "PTE conditional write failed"),
                Err(_)    => Err(PTW_No_Access(), ext_ptw),
              },
        }
      } else Err(PTW_PTE_Needs_Update(), ext_ptw),
  }

termination_measure update_and_write_pte(_, _, _, _, _, _) = 1000

// 'v is the virtual address size.
private val pt_walk : forall 'v, is_sv_or_svx4_mode('v) . (
  int('v),                      // Sv32, Sv39, Sv48, Sv57
  vpn_bits('v),                 // Virtual Page Number
  MemoryAccessType(mem_payload),  // Memory Load{Reserved}/Store{Conditional}/Atomic/InstructionFetch/CacheAccess
  Privilege,                    // Privilege
  bool,                         // mstatus.MXR
  bool,                         // do_sum
  ppn_bits('v),                 // Base PPN (`a` in the spec).
  level_range('v),              // Tree level for this recursive call (`i` in the spec).
  bool,                         // global translation
  TranslationStage,             // stage
  ext_ptw                       // ext_ptw
) -> PTW_Result('v)

// Steps 2-8 of the VATP.
function pt_walk(
  sv_width,
  vpn,
  access,
  priv,
  mxr,
  do_sum,
  pt_base,
  level,
  global,
  stage,
  ext_ptw,
) = {
  ptw_start_callback(stage, level, zero_extend(vpn), access, priv);

  let 'log_pte_size_bytes = if 'v == 32 | 'v == 34 then 2 else 3;
  let 'vpn_i_size         = pagesize_bits - 'log_pte_size_bytes;

  // Svx4 root level has 2 extra bits
  let is_svx4_root = ('v == 34 | 'v == 41 | 'v == 50 | 'v == 59) & (level == get_root_level(sv_width));

  let vpn_lo = level * vpn_i_size;
  let vpn_hi = if is_svx4_root
               then (level + 1) * vpn_i_size + 1
               else (level + 1) * vpn_i_size - 1;

  let pte_addr = (pt_base @ zeros(pagesize_bits))
               + zero_extend(vpn[vpn_hi .. vpn_lo] @ zeros('log_pte_size_bytes));

  // Convert to a physical address (34 bits for RV32, 64 bits to RV64).
  // The assertion is required so Sail knows that we can't have a
  // pte_addr of 56 bits and physaddr of 34 bits (because you can't
  // use Sv39+ on RV32).
  assert('v == 32 | 'v == 34 | xlen == 64);

  // Save before G-stage translation
  let pte_addr_original = Physaddr(zero_extend(pte_addr));

  // Translate PTE address if performing a VS-stage access
  let pte_addr : physaddr = match translate_PTE_addr(zero_extend(pte_addr), Load(PageTableEntry), stage, ext_ptw) {
    Ok(pa, _, _) => pa,
    Err((e, _))  => return Err(PTW_Implicit_Error(translationException(access, e), zero_extend(pte_addr), Load(PageTableEntry)), ext_ptw),
  };

  // Read this-level PTE from mem (Step 2 of VATP)
  match read_pte(pte_addr, 2 ^ log_pte_size_bytes) {
    Err(_)  => {
      ptw_fail_callback(stage, level, PTW_No_Access(), bits_of(pte_addr));
      Err(PTW_No_Access(), ext_ptw)
    },
    Ok(pte) => {
      ptw_step_callback(stage, level, bits_of(pte_addr), zero_extend(pte));

      let pte_flags = Mk_PTE_Flags(pte[7 .. 0]);
      // The G bit is not used in G-stage PTEs and must be ignored by hardware.
      let global = global | (stage != G_Stage & pte_flags[G] == 0b1);

      // Follow a valid non-leaf PTE to the next level of the page table
      // (Steps 3 and 4 of VATP; invalid PTEs and non-leaf PTEs at level 0
      // are rejected by check_leaf_pte() below).
      if not(pte_is_invalid(pte_flags, ext_bits_of_PTE(pte), stage)) & pte_is_non_leaf(pte_flags) & level > 0 then {
        // follow the pointer to walk next level (i.e., go to Step 2)
        let ppn = PPN_of_PTE(pte);
        pt_walk(sv_width, vpn, access, priv, mxr, do_sum, ppn, level - 1, global, stage, ext_ptw)
      }
      else
        // Steps 3 and 5-8 of VATP.
        match check_leaf_pte(sv_width, vpn, access, priv, mxr, do_sum, pte, pte_addr, level, stage, ext_ptw) {
          Err(e, ext_ptw) => Err(e, ext_ptw),
          Ok(ppn, pbmt, ext_ptw) => {
            ptw_success_callback(stage, level, zero_extend(ppn));
            // Store the original (pre-translation) PTE address for 2-stage translation in VS-Stage.
            // For G-Stage or S-Stage, pte_addr == pte_address_original.
            Ok(struct {ppn=ppn, pte=pte, pteAddr = pte_addr_original, level=level, pbmt=pbmt, global=global}, ext_ptw)
          },
        }
    }
  }
}

termination_measure pt_walk(_,_,_,_,_,_,_,level,_,_, _) = level

// ****************************************************************
// Architectural HGATP CSR
mapping clause csr_name_map = 0x680 <-> "hgatp"

function clause is_CSR_accessible(0x680, priv, _) = currentlyEnabled(Ext_H) & (priv == Machine | (priv == Supervisor & mstatus[TVM] == 0b0))
function clause is_CSR_exception_virtual(0x680, _, _) = true
function clause read_CSR(0x680) = hgatp
function clause write_CSR(0x680, value) = { hgatp = legalize_hgatp(architecture(cur_privilege), hgatp, value); Ok(hgatp) }

// Architectural VSATP CSR
mapping clause csr_name_map = 0x280 <-> "vsatp"

function clause is_CSR_accessible(0x280, _, _) = currentlyEnabled(Ext_H)
function clause read_CSR(0x280) = vsatp
function clause write_CSR(0x280, value) = { vsatp = legalize_satp(architecture(cur_privilege), true, vsatp, value); Ok(vsatp) }

// Architectural SATP CSR
private function is_satp_accessible(priv : Privilege) -> bool =
  match priv {
    Machine           => hartSupports(Ext_S),
    Supervisor        => mstatus[TVM]  == 0b0,
    VirtualSupervisor => hstatus[VTVM] == 0b0,
    User              => false,
    VirtualUser       => false,
  }

mapping clause csr_name_map = 0x180  <-> "satp"
function clause is_CSR_accessible(0x180, priv, _) = is_satp_accessible(priv)
function clause is_CSR_exception_virtual(0x180, _, _) = true
function clause read_CSR(0x180) = if inVirtualPrivilege() then vsatp else satp
function clause write_CSR(0x180, value) =
  if inVirtualPrivilege()
  then { vsatp = legalize_satp(architecture(cur_privilege), true,  vsatp, value); Ok(vsatp) }
  else { satp  = legalize_satp(architecture(Supervisor),    false, satp,  value); Ok(satp)  }

// ----------------
// Fields of SATP

private function xatp_to_ppn(stage : TranslationStage) -> bits(if xlen == 32 then 22 else 44) = {
  match stage {
    S_Stage  => if xlen == 32 then Mk_Satp32(satp)[PPN]   else Mk_Satp64(satp)[PPN],
    VS_Stage => if xlen == 32 then Mk_Satp32(vsatp)[PPN]  else Mk_Satp64(vsatp)[PPN],
    G_Stage  => if xlen == 32 then Mk_Hgatp32(hgatp)[PPN] else Mk_Hgatp64(hgatp)[PPN],
  }
}

// Return the active address translation mode for S-Stage translation.
private function satp_mode(priv : Privilege) -> SATPMode = {
  // Without S-mode enabled, satp is inactive and translation is Bare at every privilege.
  if priv == Machine | not(currentlyEnabled(Ext_S)) then Bare
  else {
    let arch = architecture(Supervisor);
    let mbits : satp_mode = match arch {
      RV64 => { assert(xlen >= 64); Mk_Satp64(satp)[Mode] },
      RV32 => 0b000 @ Mk_Satp32(satp[31..0])[Mode],
      RV128 => internal_error(__FILE__, __LINE__, "RV128 not supported"),
    };
    match satpMode_of_bits(arch, mbits) {
      Some(m) => m,
      // The model does not support modifying SXL currently so this cannot happen.
      None()  => internal_error(__FILE__, __LINE__, "invalid translation mode in satp")
    }
  }
}

// Return the active address translation mode for VS-Stage translation.
// vsatp only applies when V=1, so every caller guards on a virtual privilege
// and there is no Machine case to handle here.
private function vsatp_mode() -> SATPMode = {
  let arch = architecture(VirtualSupervisor);
  let mbits : satp_mode = match arch {
    RV64 => { assert(xlen >= 64); Mk_Satp64(vsatp)[Mode] },
    RV32 => 0b000 @ Mk_Satp32(vsatp[31..0])[Mode],
    RV128 => internal_error(__FILE__, __LINE__, "RV128 not supported"),
  };
  match satpMode_of_bits(arch, mbits) {
    Some(m) => m,
    // The model does not support modifying VSXL currently so this cannot happen.
    None()  => internal_error(__FILE__, __LINE__, "invalid translation mode in vsatp")
  }
}

// Return the active address translation mode for G-Stage translation.
// G-stage translation only happens when V=1, so as for vsatp_mode() there is
// no Machine case.
private function hgatp_mode() -> HGATPMode = {
  let arch = architecture(Supervisor);
  let mbits : hgatp_mode = match arch {
    RV64 => { assert(xlen >= 64); Mk_Hgatp64(hgatp)[Mode] },
    RV32 => 0b000 @ Mk_Hgatp32(hgatp[31..0])[Mode],
    RV128 => internal_error(__FILE__, __LINE__, "RV128 not supported"),
  };
  match hgatpMode_of_bits(arch, mbits) {
    Some(m) => m,
    // The model does not support modifying SXL currently so this cannot happen.
    None()  => internal_error(__FILE__, __LINE__, "invalid translation mode in hgatp")
  }
}

// ****************************************************************
// VA to PA translation

// Translate on TLB miss: the page-table walk (Steps 2-8 of VATP), then the
// A/D-bit update (Step 7's compare-and-swap, Step 9); if the comparison
// fails, "return to step 2" -- walk again, while `retries` allows.
private function translate_TLB_miss_retry forall 'v, is_sv_or_svx4_mode('v) . (
  retries  : nat,
  sv_width : int('v),
  asid     : asidbits,
  vmid     : vmidbits,
  base_ppn : ppn_bits('v),
  vpn      : vpn_bits('v),
  access   : MemoryAccessType(mem_payload),
  priv     : Privilege,
  mxr      : bool,
  do_sum   : bool,
  stage    : TranslationStage,
  ext_ptw  : ext_ptw,
) -> TR_Result(ppn_bits('v), PTW_Error) = {
  let initial_level = get_root_level(sv_width);

  // Step 2 of VATP occurs in pt_walk().
  let ptw_result = pt_walk(sv_width, vpn, access, priv, mxr, do_sum,
                           base_ppn, initial_level, false, stage, ext_ptw);
  match ptw_result {
    Err(f, ext_ptw) => Err(f, ext_ptw),
    Ok(struct {ppn, pte, pteAddr, level, pbmt, global}, ext_ptw) => {
      // Step 9 of VATP.
      match update_and_write_pte(sv_width, pteAddr, pte, access, stage, ext_ptw) {
        Ok(PTE_Update_Done(Some(new_pte)), ext_ptw) => {
          // Writeback the PTE (which has new A/D bits).  new_pte is the
          // value the walk read with its new A/D bits (the update compared
          // the PTE in memory with it), so the walk's ppn and pbmt
          // describe new_pte as well.
          add_to_TLB(sv_width, asid, vmid, vpn, ppn, new_pte, pteAddr, level, global, stage);
          Ok(ppn, pbmt, ext_ptw)
        },
        Ok(PTE_Update_Done(None()), ext_ptw) => {
          // There was no change in the PTE, but it still needs to be added
          // to the TLB due to the TLB miss.
          add_to_TLB(sv_width, asid, vmid, vpn, ppn, pte, pteAddr, level, global, stage);
          Ok(ppn, pbmt, ext_ptw)
        },
        Ok(PTE_Update_Retry(), ext_ptw) =>
          // The PTE was changed between the walk's read and the update's
          // exclusive read: "If the comparison fails, return to step 2."
          if retries > 0
          then translate_TLB_miss_retry(retries - 1, sv_width, asid, vmid, base_ppn, vpn, access,
                                        priv, mxr, do_sum, stage, ext_ptw)
          else internal_error(__FILE__, __LINE__, "PTE A/D update: retry bound exhausted"),
        Err(e, ext_ptw) => Err(e, ext_ptw),
      }
    },
  }
}

termination_measure translate_TLB_miss_retry(retries, _, _, _, _, _, _, _, _, _, _, _) = retries

// Translate on TLB miss (do a page-table walk)
private function translate_TLB_miss forall 'v, is_sv_or_svx4_mode('v) . (
  sv_width : int('v),
  asid     : asidbits,
  vmid     : vmidbits,
  base_ppn : ppn_bits('v),
  vpn      : vpn_bits('v),
  access   : MemoryAccessType(mem_payload),
  priv     : Privilege,
  mxr      : bool,
  do_sum   : bool,
  stage    : TranslationStage,
  ext_ptw  : ext_ptw,
) -> TR_Result(ppn_bits('v), PTW_Error) =
  translate_TLB_miss_retry(pte_update_max_retries, sv_width, asid, vmid, base_ppn, vpn, access,
                           priv, mxr, do_sum, stage, ext_ptw)

// This function can be ignored on first reading since TLBs are not
// part of RISC-V architecture spec (see TLB NOTE above).
// Translate on TLB hit, and maintenance of PTE in TLB
private function translate_TLB_hit forall 'v, is_sv_or_svx4_mode('v) . (
  sv_width  : int('v),
  asid      : asidbits,
  vmid      : vmidbits,
  base_ppn  : ppn_bits('v),
  vpn       : vpn_bits('v),
  access    : MemoryAccessType(mem_payload),
  priv      : Privilege,
  mxr       : bool,
  do_sum    : bool,
  tlb_index : tlb_index_range,
  ent       : TLB_Entry,
  stage     : TranslationStage,
  ext_ptw   : ext_ptw,
) -> TR_Result(ppn_bits('v), PTW_Error) = {

  let 'pte_size = if 'v == 32 | 'v == 34 then 4 else 8;
  let pte       = tlb_get_pte(pte_size, ent);  // Step 2 of VATP.
  let ext_pte   = ext_bits_of_PTE(pte);
  let pte_flags = Mk_PTE_Flags(pte[7 .. 0]);
  let pte_check = check_PTE_permission(access, priv, stage, mxr, do_sum, pte_flags,
                                       ext_pte, ext_ptw);
  match pte_check {
    PTE_Check_Failure(ext_ptw, pte_failure) =>
      Err(ext_get_ptw_error(pte_failure), ext_ptw),
    PTE_Check_Success(ext_ptw) => {
      // Step 9 of VATP.  The A/D-bit update is performed atomically on
      // the PTE in memory (see update_and_write_pte()), compared with the
      // cached copy; if the in-memory PTE differs from the cached copy
      // (it was modified since this TLB entry was filled) the copy is
      // stale, and the translation walks the page table instead (the walk
      // refills this TLB slot) rather than writing a stale copy back.
      match update_and_write_pte(sv_width, ent.pteAddr, pte, access, stage, ext_ptw) {
        Ok(PTE_Update_Done(Some(pte)), ext_ptw) => {
          // Writeback the PTE (which has new A/D bits).
          write_TLB(tlb_index, tlb_set_pte(ent, pte : bits('pte_size * 8)));
          Ok(tlb_get_ppn(sv_width, ent, vpn), tlb_get_pbmt(ent), ext_ptw)
        },
        Ok(PTE_Update_Done(None()), ext_ptw) => Ok(tlb_get_ppn(sv_width, ent, vpn), tlb_get_pbmt(ent), ext_ptw),
        Ok(PTE_Update_Retry(), ext_ptw) =>
          translate_TLB_miss(sv_width, asid, vmid, base_ppn, vpn, access, priv,
                             mxr, do_sum, stage, ext_ptw),
        Err(e, ext_ptw) => Err(e, ext_ptw),
      }
    },
  }
}

// Mapping the SATPMode to the width integer. Note there is also SvBare
// and it's an error to call this with SvBare.
mapping satp_mode_width : SATPMode <-> {32, 39, 48, 57} = {
  Sv32 <-> 32,
  Sv39 <-> 39,
  Sv48 <-> 48,
  Sv57 <-> 57,
}

mapping hgatp_mode_width : HGATPMode <-> {34, 41, 50, 59} = {
  Sv32x4 <-> 34,
  Sv39x4 <-> 41,
  Sv48x4 <-> 50,
  Sv57x4 <-> 59,
}

private function translate forall 'v, is_sv_or_svx4_mode('v) . (
  sv_width : int('v),
  asid     : asidbits,
  vmid     : vmidbits,
  base_ppn : ppn_bits('v),
  vpn      : vpn_bits('v),
  access   : MemoryAccessType(mem_payload),
  priv     : Privilege,
  mxr      : bool,
  do_sum   : bool,
  stage    : TranslationStage,
  ext_ptw  : ext_ptw,
) -> TR_Result(ppn_bits('v), PTW_Error) = {
  // On first reading, assume lookup_TLB returns None(), since TLBs
  // are not part of RISC-V architecture spec (see TLB NOTE above)
  match lookup_TLB(sv_width, asid, vmid, vpn, stage) {
    Some(index, ent) => translate_TLB_hit(sv_width, asid, vmid, base_ppn, vpn, access, priv,
                                          mxr, do_sum, index, ent, stage, ext_ptw),
    None()           => {
      generate_event(EV_TLB_MISS);
      translate_TLB_miss(sv_width, asid, vmid, base_ppn, vpn, access, priv,
                                           mxr, do_sum, stage, ext_ptw)
    },
  }
}

private function get_mxr(stage : TranslationStage) -> bool =
  if stage == VS_Stage then mstatus[MXR] == 0b1 | vsstatus[MXR] == 0b1 else mstatus[MXR] == 0b1

private function get_sum(stage : TranslationStage) -> bool =
  if stage == VS_Stage then vsstatus[SUM] == 0b1 else mstatus[SUM]  == 0b1

private val translate_validate_address : forall 'n 'v, is_sv_or_svx4_mode('v) & 'n >= 'v. (bits('n), int('v), TranslationStage) -> result(bits('v), PTW_Error)
function translate_validate_address(addr, sv_width, stage) = {
  let truncated           = truncate(addr, sv_width);
  let extended : bits('n) = if stage == G_Stage then zero_extend(truncated) else sign_extend(truncated);
  if addr == extended
  then Ok(truncated)
  else Err(PTW_Invalid_Addr())
}

private function translate_stage forall 'n 'v, is_sv_or_svx4_mode('v) & 'n >= 'v. (
  sv_width : int('v),
  addr     : bits('n),
  access   : MemoryAccessType(mem_payload),
  priv     : Privilege,
  stage    : TranslationStage
) -> TR_Result(gpabits, PTW_Error) = {
  match translate_validate_address(addr, sv_width, stage) {
    Err(e)         => Err(e, init_ext_ptw),
    Ok(valid_addr) => {
      assert((xlen == 32 & (sv_width == 32 | sv_width == 34)) |
             (xlen == 64 &  sv_width != 32 & sv_width != 34));
      let asid     = get_asid(stage);
      let vmid     = get_vmid(stage);
      let mxr      = get_mxr(stage);
      let do_sum   = get_sum(stage);
      let vpn      = valid_addr[sv_width - 1 .. pagesize_bits];

      // Step 1 of VATP.
      let base_ppn = xatp_to_ppn(stage);
      let eff_priv = if stage == G_Stage then User else priv;
      let res = translate(sv_width, asid, vmid,
                          base_ppn, vpn, access,
                          eff_priv, mxr, do_sum,
                          stage, init_ext_ptw);
      match res {
        Ok(ppn, pbmt, ext_ptw) => Ok(zero_extend(ppn @ valid_addr[pagesize_bits - 1 .. 0]), pbmt, ext_ptw),
        Err(f, ext_ptw)  => Err(f, ext_ptw)
      }
    }
  }
}

// Wrapper for VS/S-stage translation (takes virtaddr)
private function translate_vs_stage(
  vaddr  : virtaddr,
  access : MemoryAccessType(mem_payload),
  p      : Privilege
) -> TR_Result(gpabits, PTW_Error) = {
  let (stage, mode) = if p == VirtualSupervisor | p == VirtualUser
                      then (VS_Stage, vsatp_mode())
                      else (S_Stage, satp_mode(p));
  if mode == Bare then return Ok(zero_extend(bits_of(vaddr)), PBMT_PMA, init_ext_ptw);
  let sv_width = satp_mode_width(mode);
  assert(sv_width <= xlen);
  translate_stage(sv_width, bits_of(vaddr), access, p, stage)
}

// Wrapper for G-stage translation (takes physaddr).
// G-stage permission checking always uses User, so no privilege is taken.
function translate_g_stage(
  gpa    : gpabits,
  access : MemoryAccessType(mem_payload),
) -> TR_Result(gpabits, PTW_Error) = {
  let mode = hgatp_mode();
  if mode == HBare then return Ok(gpa, PBMT_PMA, init_ext_ptw);
  let sv_width = hgatp_mode_width(mode);
  assert(sv_width <= gpalen);
  translate_stage(sv_width, gpa, access, User, G_Stage)
}

// This option tells Sail to unroll the multi-stage translation for theorem prover
// targets, so that we only need a termination measure for the main pt_walk function.
// This is a good point to break the cycle because the G-stage translation cannot
// recursively translate PTE addresses again.
$option --const-prop-mutrec translate_g_stage

// There are three cases in which an access can fail during two-stage address
// translation, shown here with the single-stage case for comparison. Without
// H there is no htval register at all. With H it is zero here because the
// fault is not a guest-page fault.
//
// Case      Exception                        stval htval      Who Handles
// Single    S-Stage  fail Regular page fault VA    0 (if H)   Supervisor
// Implicit  VS-Stage fail Guest-page fault   GVA   PTE's GPA  Hypervisor
// Explicit  VS-Stage fail Regular page fault GVA   0          Virtual Supervisor
// Explicit  G-Stage  fail Guest-page fault   GVA   GPA        Hypervisor
//
// We save all the information in ExceptionContext in order to set the appropriate
// CSR's according to the effective privilege mode and given faulting stage.
private function build_exception_context(
  gva       : virtaddr,
  gpa       : gpabits,
  access    : MemoryAccessType(mem_payload),
  ptw_error : PTW_Error,
  stage     : TranslationStage
) -> ExceptionContext = {
  match stage {
    S_Stage => {
      struct {
        trap  = translationException(access, ptw_error),
        excinfo_is_gva = false,
        excinfo  = zero_extend(bits_of(gva)),
        excinfo2 = None(),
        excinst  = None(),
        excinst_offset = zeros(),
        ext      = None(),
      }
    },
    VS_Stage => {
      match ptw_error {
        PTW_Implicit_Error(e, pte_addr, access) => {
          let is_write : bool = match access {
            Store(_) => true,
            Load(_)  => false,
            _        => internal_error(__FILE__, __LINE__, "invalid memory access type"),
          };
          let trap = convertToGuestException(e);
          let tval2 = g_stage_tval2_exception_value(trap, pte_addr);
          struct {
            trap     = trap,
            excinfo_is_gva = true,
            excinfo  = zero_extend(bits_of(gva)),
            excinfo2 = Some(tval2),
            // NOTE: "when a guest-page fault is caused by an implicit
            // memory access during VS-stage address translation and htval/mtval2 is nonzero,
            // a pseudoinstruction MUST be written to htinst/mtinst (zero is not allowed).
            // The pseudoinstruction encodes whether the implicit access was a read (PTE fetch)
            // or a write (hardware A/D bit update in VS-level page tables)."
            //
            // 0x00002000 = 32-bit read  (hardware was reading a PTE)
            // 0x00002020 = 32-bit write (hardware was updating A/D bits)
            // 0x00003000 = 64-bit read
            // 0x00003020 = 64-bit write
            excinst = if tval2 != zeros()
                      then Some(if xlen == 32
                                then if is_write then zero_extend(0x00002020) else zero_extend(0x00002000)
                                else if is_write then zero_extend(0x00003020) else zero_extend(0x00003000))
                      else None(),
            excinst_offset = zeros(),
            ext = None(),
          }
        },
        _ => {
          // Explicit VS-Stage failure
          struct {
            trap  = translationException(access, ptw_error),
            excinfo_is_gva = true,
            excinfo  = zero_extend(bits_of(gva)),
            excinfo2 = Some(zeros()),
            excinst  = None(),
            excinst_offset = zeros(),
            ext      = None(),
          }
        }
      }
    },
    G_Stage => {
      let trap = convertToGuestException(translationException(access, ptw_error));
      struct {
        trap     = trap,
        excinfo_is_gva = true,
        excinfo  = zero_extend(bits_of(gva)),
        excinfo2 = Some(g_stage_tval2_exception_value(trap, gpa)),
        excinst  = None(),
        excinst_offset = zeros(),
        ext      = None(),
      }
    }
  }
}

// Checks if an access is Svukte-qualified under the Svukte extension. If the extension is not
// enabled, the UKTE bit of senvcfg is read-only-zero, so accesses will never be qualified.
function is_svukte_qualified(
  access   : MemoryAccessType(mem_payload),
  eff_priv : Privilege,
  mode     : SATPMode
) -> bool = {
  // Only instruction fetches and explicit accesses can be Svukte-qualified.
  let qualified_access_type = access == InstructionFetch() | not(is_implicit_access(access));

  let ukte_enable = if eff_priv == VirtualUser & cur_privilege == User
                    then hstatus[HUKTE]
                    else read_senvcfg()[UKTE];

  mode != Bare &
    (eff_priv == User | eff_priv == VirtualUser) &
    qualified_access_type &
    ukte_enable == 0b1
}

// Top-level addr-translation function
// PUBLIC: invoked from load-store utils, instr-fetch, atomics and CBOs, etc.
function translateAddr(
  gva    : virtaddr,
  access : MemoryAccessType(mem_payload),
  p      : Privilege
) -> TR_Result(physaddr, ExceptionContext) = {

  // True if the effective privilege is virtual (VS/VU), as opposed to
  // the hart's current privilege. Used by hlv/hsv which execute in
  // HS or M mode but emulate a virtual-mode memory access.
  let is_virtual_access = p == VirtualSupervisor | p == VirtualUser;

  let mode = if is_virtual_access then vsatp_mode() else satp_mode(p);

  // "If satp.MODE (or vsatp.MODE when V=1) is set to Bare and the
  // effective privilege mode is less than M, shadow stack
  // instructions raise a store/AMO access-fault exception."
  // "Use of Zicfiss in M-mode is not supported."
  if is_shadow_stack_access(access) then {
    if ((mode == Bare & p != Machine) | p == Machine) then
    // build_exception_context expects ptw_error, as a work around we manually
    // construct the exception context.
    return Err(struct {
      trap  = E_SAMO_Access_Fault(),
      excinfo_is_gva = is_virtual_access,
      excinfo  = zero_extend(bits_of(gva)),
      excinfo2 = None(),
      excinst  = None(),
      excinst_offset = zeros(),
      ext      = None(),
    }, init_ext_ptw);
  };

  // Svukte-qualified accesses are invalid if their virtual address is >= 2^(SXLEN-1).
  // This check uses this model's assumption that SXLEN == XLEN, and would need reworking if that
  // assumption were to change in the future.
  if is_svukte_qualified(access, p, mode) & (bits_of(gva)[xlen - 1] == bitone) then {
    return Err(make_exception_context(translationException(access, PTW_Invalid_Addr()), zero_extend(bits_of(gva)), is_virtual_access, None()), init_ext_ptw);
  };

  // First stage: GVA -> GPA (or VA -> PA for single-stage)
  match translate_vs_stage(gva, access, p) {
    Ok(gpa, vs_pbmt, ext_ptw) => {
      if not(is_virtual_access) then {
        return Ok(Physaddr(gpa), vs_pbmt, ext_ptw)
      };
      // Second stage: GPA -> SPA
      match translate_g_stage(gpa, access) {
        Ok(spa, g_pbmt, ext_ptw)  => {
          // A non-zero G-stage PBMT overrides the PMAs to give the intermediate
          // attributes, then a non-zero VS-stage PBMT overrides those in turn.
          let pbmt = if vs_pbmt != PBMT_PMA then vs_pbmt else g_pbmt;
          Ok(Physaddr(spa), pbmt, ext_ptw)
        },
        // G-Stage failure
        Err(ptw_error, ext_ptw) => Err(build_exception_context(gva, gpa, access, ptw_error, G_Stage), ext_ptw)
      }
    },
    // S/VS-stage failure
    Err(ptw_error, ext_ptw) => Err(build_exception_context(gva, zeros(), access, ptw_error, if is_virtual_access then VS_Stage else S_Stage), ext_ptw)
  }
}

// ****************************************************************
// Initialize Virtual Memory state

// PUBLIC: invoked from reset() [postlude/model.sail]
function reset_vmem() -> unit = {
  // "If the hypervisor extension is implemented, the hgatp.MODE and vsatp.MODE fields are reset to 0."
  vsatp = zeros();
  hgatp = zeros();

  reset_TLB()
}

// ****************************************************************
