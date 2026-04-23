use core::cmp::Ordering;
use rand::Rng;
use crunchy::unroll;

use byteorder::{BigEndian, ByteOrder};

/// 256-bit, stack allocated biginteger for use in prime field
/// arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct U256(pub [u128; 2]);

impl From<[u64; 4]> for U256 {
    fn from(d: [u64; 4]) -> Self {
        let mut a = [0u128; 2];
        a[0] = (d[1] as u128) << 64 | d[0] as u128;
        a[1] = (d[3] as u128) << 64 | d[2] as u128;
        U256(a)
    }
}

impl From<u64> for U256 {
    fn from(d: u64) -> Self {
        U256::from([d, 0, 0, 0])
    }
}

/// 512-bit, stack allocated biginteger for use in extension
/// field serialization and scalar interpretation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct U512(pub [u128; 4]);

impl From<[u64; 8]> for U512 {
    fn from(d: [u64; 8]) -> Self {
        let mut a = [0u128; 4];
        a[0] = (d[1] as u128) << 64 | d[0] as u128;
        a[1] = (d[3] as u128) << 64 | d[2] as u128;
        a[2] = (d[5] as u128) << 64 | d[4] as u128;
        a[3] = (d[7] as u128) << 64 | d[6] as u128;
        U512(a)
    }
}

impl U512 {
    /// Multiplies c1 by modulo, adds c0.
    pub fn new(c1: &U256, c0: &U256, modulo: &U256) -> U512 {
        let mut res = [0; 4];

        debug_assert_eq!(c1.0.len(), 2);
        unroll! {
            for i in 0..2 {
                mac_digit(i, &mut res, &modulo.0, c1.0[i]);
            }
        }

        let mut carry = 0;

        debug_assert_eq!(res.len(), 4);
        unroll! {
            for i in 0..2 {
                res[i] = adc(res[i], c0.0[i], &mut carry);
            }
        }

        unroll! {
            for i in 0..2 {
                let (a1, a0) = split_u128(res[i + 2]);
                let (c, r0) = split_u128(a0 + carry);
                let (c, r1) = split_u128(a1 + c);
                carry = c;

                res[i + 2] = combine_u128(r1, r0);
            }
        }

        debug_assert!(0 == carry);

        U512(res)
    }

     pub fn from_slice(s: &[u8]) -> Result<U512, Error> {
        if s.len() != 64 {
            return Err(Error::InvalidLength {
                expected: 32,
                actual: s.len(),
            });
        }

        let mut n = [0; 4];
        for (l, i) in (0..4).rev().zip((0..4).map(|i| i * 16)) {
            n[l] = BigEndian::read_u128(&s[i..]);
        }

        Ok(U512(n))
    }

    /// Get a random U512
    pub fn random<R: Rng>(rng: &mut R) -> U512 {
        U512(rng.gen())
    }

    pub fn get_bit(&self, n: usize) -> Option<bool> {
        if n >= 512 {
            None
        } else {
            let part = n / 128;
            let bit = n - (128 * part);

            Some(self.0[part] & (1 << bit) > 0)
        }
    }

    /// Divides self by modulo, returning remainder and, if
    /// possible, a quotient smaller than the modulus.
    pub fn divrem(&self, modulo: &U256) -> (Option<U256>, U256) {
        let mut q = Some(U256::zero());
        let mut r = U256::zero();

        for i in (0..512).rev() {
            // NB: modulo's first two bits are always unset
            // so this will never destroy information
            mul2(&mut r.0);
            assert!(r.set_bit(0, self.get_bit(i).unwrap()));
            if &r >= modulo {
                sub_noborrow(&mut r.0, &modulo.0);
                if q.is_some() && !q.as_mut().unwrap().set_bit(i, true) {
                    q = None
                }
            }
        }

        if q.is_some() && (q.as_ref().unwrap() >= modulo) {
            (None, r)
        } else {
            (q, r)
        }
    }

    pub fn interpret(buf: &[u8; 64]) -> U512 {
        let mut n = [0; 4];
        for (l, i) in (0..4).rev().zip((0..4).map(|i| i * 16)) {
            n[l] = BigEndian::read_u128(&buf[i..]);
        }

        U512(n)
    }
}

impl Ord for U512 {
    #[inline]
    fn cmp(&self, other: &U512) -> Ordering {
        for (a, b) in self.0.iter().zip(other.0.iter()).rev() {
            if *a < *b {
                return Ordering::Less;
            } else if *a > *b {
                return Ordering::Greater;
            }
        }

        return Ordering::Equal;
    }
}

impl PartialOrd for U512 {
    #[inline]
    fn partial_cmp(&self, other: &U512) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &U256) -> Ordering {
        for (a, b) in self.0.iter().zip(other.0.iter()).rev() {
            if *a < *b {
                return Ordering::Less;
            } else if *a > *b {
                return Ordering::Greater;
            }
        }

        return Ordering::Equal;
    }
}

impl PartialOrd for U256 {
    #[inline]
    fn partial_cmp(&self, other: &U256) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// U256/U512 errors
#[derive(Debug)]
pub enum Error {
    InvalidLength { expected: usize, actual: usize },
}

impl U256 {
    /// Initialize U256 from slice of bytes (big endian)
    pub fn from_slice(s: &[u8]) -> Result<U256, Error> {
        if s.len() != 32 {
            return Err(Error::InvalidLength {
                expected: 32,
                actual: s.len(),
            });
        }

        let mut n = [0; 2];
        for (l, i) in (0..2).rev().zip((0..2).map(|i| i * 16)) {
            n[l] = BigEndian::read_u128(&s[i..]);
        }

        Ok(U256(n))
    }

    pub fn to_big_endian(&self, s: &mut [u8]) -> Result<(), Error> {
        if s.len() != 32 {
            return Err(Error::InvalidLength {
                expected: 32,
                actual: s.len(),
            });
        }

        for (l, i) in (0..2).rev().zip((0..2).map(|i| i * 16)) {
            BigEndian::write_u128(&mut s[i..], self.0[l]);
        }

        Ok(())
    }

    #[inline]
    pub fn zero() -> U256 {
        U256([0, 0])
    }

    #[inline]
    pub fn one() -> U256 {
        U256([1, 0])
    }

    /// Produce a random number (mod `modulo`)
    pub fn random<R: Rng>(rng: &mut R, modulo: &U256) -> U256 {
        U512::random(rng).divrem(modulo).1
    }

    pub fn is_zero(&self) -> bool {
        self.0[0] == 0 && self.0[1] == 0
    }

    pub fn set_bit(&mut self, n: usize, to: bool) -> bool {
        if n >= 256 {
            false
        } else {
            let part = n / 128;
            let bit = n - (128 * part);

            if to {
                self.0[part] |= 1 << bit;
            } else {
                self.0[part] &= !(1 << bit);
            }

            true
        }
    }

    pub fn get_bit(&self, n: usize) -> Option<bool> {
        if n >= 256 {
            None
        } else {
            let part = n / 128;
            let bit = n - (128 * part);

            Some(self.0[part] & (1 << bit) > 0)
        }
    }

    /// Add `other` to `self` (mod `modulo`)
    pub fn add(&mut self, other: &U256, modulo: &U256) {
        add_nocarry(&mut self.0, &other.0);

        if *self >= *modulo {
            sub_noborrow(&mut self.0, &modulo.0);
        }
    }

    /// Subtract `other` from `self` (mod `modulo`)
    pub fn sub(&mut self, other: &U256, modulo: &U256) {
        if *self < *other {
            add_nocarry(&mut self.0, &modulo.0);
        }

        sub_noborrow(&mut self.0, &other.0);
    }

    /// Multiply `self` by `other` (mod `modulo`) via the Montgomery
    /// multiplication method.
    pub fn mul(&mut self, other: &U256, modulo: &U256, inv: u128) {
        mul_reduce(&mut self.0, &other.0, &modulo.0, inv);

        if *self >= *modulo {
            sub_noborrow(&mut self.0, &modulo.0);
        }
    }

    /// Turn `self` into its additive inverse (mod `modulo`)
    pub fn neg(&mut self, modulo: &U256) {
        if *self > Self::zero() {
            let mut tmp = modulo.0;
            sub_noborrow(&mut tmp, &self.0);

            self.0 = tmp;
        }
    }

    #[inline]
    pub fn is_even(&self) -> bool {
        self.0[0] & 1 == 0
    }

    /// Turn `self` into its multiplicative inverse (mod `modulo`)
    pub fn invert(&mut self, modulo: &U256) {
        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let mut u = *self;
        let mut v = *modulo;
        let mut b = U256::one();
        let mut c = U256::zero();

        while u != U256::one() && v != U256::one() {
            while u.is_even() {
                div2(&mut u.0);

                if b.is_even() {
                    div2(&mut b.0);
                } else {
                    add_nocarry(&mut b.0, &modulo.0);
                    div2(&mut b.0);
                }
            }
            while v.is_even() {
                div2(&mut v.0);

                if c.is_even() {
                    div2(&mut c.0);
                } else {
                    add_nocarry(&mut c.0, &modulo.0);
                    div2(&mut c.0);
                }
            }

            if u >= v {
                sub_noborrow(&mut u.0, &v.0);
                b.sub(&c, modulo);
            } else {
                sub_noborrow(&mut v.0, &u.0);
                c.sub(&b, modulo);
            }
        }

        if u == U256::one() {
            self.0 = b.0;
        } else {
            self.0 = c.0;
        }
    }

    /// Return an Iterator<Item=bool> over all bits from
    /// MSB to LSB.
    pub fn bits(&self) -> BitIterator {
        BitIterator { int: &self, n: 256 }
    }
}

pub struct BitIterator<'a> {
    int: &'a U256,
    n: usize,
}

impl<'a> Iterator for BitIterator<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.n == 0 {
            None
        } else {
            self.n -= 1;

            self.int.get_bit(self.n)
        }
    }
}

/// Divide by two
#[inline]
fn div2(a: &mut [u128; 2]) {
    let tmp = a[1] << 127;
    a[1] >>= 1;
    a[0] >>= 1;
    a[0] |= tmp;
}

/// Multiply by two
#[inline]
fn mul2(a: &mut [u128; 2]) {
    let tmp = a[0] >> 127;
    a[0] <<= 1;
    a[1] <<= 1;
    a[1] |= tmp;
}

#[inline(always)]
fn split_u128(i: u128) -> (u128, u128) {
    (i >> 64, i & 0xFFFFFFFFFFFFFFFF)
}

#[inline(always)]
fn combine_u128(hi: u128, lo: u128) -> u128 {
    (hi << 64) | lo
}

#[inline]
fn adc(a: u128, b: u128, carry: &mut u128) -> u128 {
    let (a1, a0) = split_u128(a);
    let (b1, b0) = split_u128(b);
    let (c, r0) = split_u128(a0 + b0 + *carry);
    let (c, r1) = split_u128(a1 + b1 + c);
    *carry = c;

    combine_u128(r1, r0)
}

#[inline]
fn add_nocarry(a: &mut [u128; 2], b: &[u128; 2]) {
    let mut carry = 0;

    for (a, b) in a.into_iter().zip(b.iter()) {
        *a = adc(*a, *b, &mut carry);
    }

    debug_assert!(0 == carry);
}

#[inline]
fn sub_noborrow(a: &mut [u128; 2], b: &[u128; 2]) {
    #[inline]
    fn sbb(a: u128, b: u128, borrow: &mut u128) -> u128 {
        let (a1, a0) = split_u128(a);
        let (b1, b0) = split_u128(b);
        let (b, r0) = split_u128((1 << 64) + a0 - b0 - *borrow);
        let (b, r1) = split_u128((1 << 64) + a1 - b1 - ((b == 0) as u128));

        *borrow = (b == 0) as u128;

        combine_u128(r1, r0)
    }

    let mut borrow = 0;

    for (a, b) in a.into_iter().zip(b.iter()) {
        *a = sbb(*a, *b, &mut borrow);
    }

    debug_assert!(0 == borrow);
}

// TODO: Make `from_index` a const param
#[inline(always)]
fn mac_digit(from_index: usize, acc: &mut [u128; 4], b: &[u128; 2], c: u128) {
    #[inline]
    fn mac_with_carry(a: u128, b: u128, c: u128, carry: &mut u128) -> u128 {
        let (b_hi, b_lo) = split_u128(b);
        let (c_hi, c_lo) = split_u128(c);

        let (a_hi, a_lo) = split_u128(a);
        let (carry_hi, carry_lo) = split_u128(*carry);
        let (x_hi, x_lo) = split_u128(b_lo * c_lo + a_lo + carry_lo);
        let (y_hi, y_lo) = split_u128(b_lo * c_hi);
        let (z_hi, z_lo) = split_u128(b_hi * c_lo);
        // Brackets to allow better ILP
        let (r_hi, r_lo) = split_u128((x_hi + y_lo) + (z_lo + a_hi) + carry_hi);

        *carry = (b_hi * c_hi) + r_hi + y_hi + z_hi;

        combine_u128(r_lo, x_lo)
    }

    if c == 0 {
        return;
    }

    let mut carry = 0;

    debug_assert_eq!(acc.len(), 4);
    unroll! {
        for i in 0..2 {
            let a_index = i + from_index;
            acc[a_index] = mac_with_carry(acc[a_index], b[i], c, &mut carry);
        }
    }
    unroll! {
        for i in 0..2 {
            let a_index = i + from_index + 2;
            if a_index < 4 {
                let (a_hi, a_lo) = split_u128(acc[a_index]);
                let (carry_hi, carry_lo) = split_u128(carry);
                let (x_hi, x_lo) = split_u128(a_lo + carry_lo);
                let (r_hi, r_lo) = split_u128(x_hi + a_hi + carry_hi);

                carry = r_hi;

                acc[a_index] = combine_u128(r_lo, x_lo);
            }
        }
    }

    debug_assert!(carry == 0);
}

#[inline]
fn mul_reduce(this: &mut [u128; 2], by: &[u128; 2], modulus: &[u128; 2], inv: u128) {
    #[cfg(all(feature = "cortex-m33-asm", target_arch = "arm"))]
    mul_reduce_asm(this, by, modulus, inv);
    #[cfg(not(all(feature = "cortex-m33-asm", target_arch = "arm")))]
    mul_reduce_rust(this, by, modulus, inv);
}

// Kept as the portable reference and the default on non-arm targets.
// On firmware with `cortex-m33-asm` enabled, the dispatcher calls the asm
// version instead, so this body becomes unused in that build.
#[allow(dead_code)]
#[inline(always)]
fn mul_reduce_rust(this: &mut [u128; 2], by: &[u128; 2], modulus: &[u128; 2], inv: u128) {
    // The Montgomery reduction here is based on Algorithm 14.32 in
    // Handbook of Applied Cryptography
    // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

    let mut res = [0; 2 * 2];
    unroll! {
        for i in 0..2 {
            mac_digit(i, &mut res, by, this[i]);
        }
    }

    unroll! {
        for i in 0..2 {
            let k = inv.wrapping_mul(res[i]);
            mac_digit(i, &mut res, modulus, k);
        }
    }

    this.copy_from_slice(&res[2..]);
}

#[cfg(all(feature = "cortex-m33-asm", target_arch = "arm"))]
fn mul_reduce_asm(this: &mut [u128; 2], by: &[u128; 2], modulus: &[u128; 2], inv: u128) {
    extern "C" {
        fn mul_reduce_armv8m(
            this: *mut u32,
            by: *const u32,
            modulus: *const u32,
            inv_low: u32,
        );
    }
    // [u128; 2] is 32 bytes with 16-byte alignment, which is a valid view
    // as 8 × u32 in the same memory (u32 needs 4-byte alignment). The asm
    // function reads/writes exactly that view and only uses `inv` mod 2^32.
    let inv_low = inv as u32;
    // SAFETY: the three slices have 8 × u32 of valid memory each (32 bytes,
    // 16-byte aligned). `mul_reduce_armv8m` reads 8 u32s from `by` and
    // `modulus`, reads and writes 8 u32s at `this`, and uses `inv_low` as
    // a scalar. No aliasing between the three buffers (caller-enforced by
    // passing distinct `&mut`/`&` references).
    unsafe {
        mul_reduce_armv8m(
            this.as_mut_ptr().cast::<u32>(),
            by.as_ptr().cast::<u32>(),
            modulus.as_ptr().cast::<u32>(),
            inv_low,
        );
    }
}

// ARMv8-M hand-written Montgomery reduction.
//
// Algorithm: Separated Operand Scanning (SOS) on 8 × u32 limbs. Phase 1 is
// an 8×8 schoolbook a*b written into a 16-word accumulator `t`. Phase 2 is
// 8 Montgomery-reduction rows, each multiplying the modulus by m = t[i] *
// inv_low (mod 2^32) and adding into t[i..i+8] with full carry propagation
// through the top of t. After Phase 2, t[8..16] holds the result.
//
// The inner step maps exactly onto UMAAL: `t[k] := a_i*b_j + t[k] + carry`
// with the new carry in one register and the low 32 bits in another.
//
// Register map (AAPCS32) — tuned to keep `by` live in registers across all
// of Phase 1 and `modulus` live across all of Phase 2, avoiding the 64+64
// redundant LDRs the first pass spent on operand reloads:
//   r0    = this ptr            (kept)
//   r1    = a[i] (phase 1) / m (phase 2) — row-multiplier scratch
//   r2    = modulus ptr         (kept, used to reload r4..r11 between phases)
//   r3    = inv_low             (kept)
//   r4..r11 = b[0..7] in phase 1, p[0..7] in phase 2
//   r12   = t[i+j] (UMAAL RdLo) scratch
//   lr    = carry (UMAAL RdHi) scratch, pre-saved by push frame
//   sp    = 64-byte t buffer
//
// Section placement: `.ram_text.*` so the firmware's pre_init copy moves
// this function into SRAM at boot. Running from RAM avoids XIP-cache
// pressure on the rest of the hot pairing path, wich still runs from flash.
#[cfg(all(feature = "cortex-m33-asm", target_arch = "arm"))]
core::arch::global_asm!(
    r#"
    .syntax unified
    .thumb
    .cpu cortex-m33

    // UMAAL step using the row multiplier in r1 and a preloaded column
    // operand register (b[j] in phase 1, p[j] in phase 2). tij lives in r12,
    // carry in lr.
    .macro UMAAL_STEP col_reg, tij_off
        ldr     r12, [sp, #\tij_off]
        umaal   r12, lr, r1, \col_reg
        str     r12, [sp, #\tij_off]
    .endm

    // Phase 1 outer loop row i: a[i] -> r1, then 8 UMAALs against b[0..7]
    // which are held in r4..r11 for the duration of Phase 1.
    .macro MUL_ROW_REG i
        ldr     r1, [r0, #4*\i]
        movs    lr, #0
        UMAAL_STEP r4,  4*\i
        UMAAL_STEP r5,  4*\i + 4
        UMAAL_STEP r6,  4*\i + 8
        UMAAL_STEP r7,  4*\i + 12
        UMAAL_STEP r8,  4*\i + 16
        UMAAL_STEP r9,  4*\i + 20
        UMAAL_STEP r10, 4*\i + 24
        UMAAL_STEP r11, 4*\i + 28
        str     lr, [sp, #4*\i + 32]
    .endm

    // Specialized Phase 1 row 0. t[0..7] is uninitialized on entry; each
    // UMAAL's RdLo is seeded with `movs r12, #0` rather than a load from
    // stack. Saves 8 LDRs in row 0 and removes the need to zero-init
    // t[0..7] upfront.
    .macro MUL_ROW_ZERO
        ldr     r1, [r0, #0]
        movs    lr, #0
        movs    r12, #0
        umaal   r12, lr, r1, r4
        str     r12, [sp, #0]
        movs    r12, #0
        umaal   r12, lr, r1, r5
        str     r12, [sp, #4]
        movs    r12, #0
        umaal   r12, lr, r1, r6
        str     r12, [sp, #8]
        movs    r12, #0
        umaal   r12, lr, r1, r7
        str     r12, [sp, #12]
        movs    r12, #0
        umaal   r12, lr, r1, r8
        str     r12, [sp, #16]
        movs    r12, #0
        umaal   r12, lr, r1, r9
        str     r12, [sp, #20]
        movs    r12, #0
        umaal   r12, lr, r1, r10
        str     r12, [sp, #24]
        movs    r12, #0
        umaal   r12, lr, r1, r11
        str     r12, [sp, #28]
        str     lr, [sp, #32]
    .endm

    // Phase 2 inner for reduction row i: m = t[i] * inv_low in r1, then 8
    // UMAALs against p[0..7] in r4..r11. Final carry in lr for propagation.
    .macro REDUCE_INNER_REG i
        ldr     r1, [sp, #4*\i]
        mul     r1, r1, r3
        movs    lr, #0
        UMAAL_STEP r4,  4*\i
        UMAAL_STEP r5,  4*\i + 4
        UMAAL_STEP r6,  4*\i + 8
        UMAAL_STEP r7,  4*\i + 12
        UMAAL_STEP r8,  4*\i + 16
        UMAAL_STEP r9,  4*\i + 20
        UMAAL_STEP r10, 4*\i + 24
        UMAAL_STEP r11, 4*\i + 28
    .endm

    // Adds lr (final carry) into t[at_off], setting flags for the ADC chain.
    .macro ADD_CARRY_AT at_off
        ldr     r1, [sp, #\at_off]
        adds    r1, r1, lr
        str     r1, [sp, #\at_off]
    .endm

    // Propagates the carry flag one word up the accumulator.
    .macro ADC_ZERO_AT at_off
        ldr     r1, [sp, #\at_off]
        adcs    r1, r1, #0
        str     r1, [sp, #\at_off]
    .endm

    .section .ram_text.mul_reduce_armv8m,"ax",%progbits
    .align 2
    .global mul_reduce_armv8m
    .type mul_reduce_armv8m, %function
    .thumb_func
mul_reduce_armv8m:
    push    {{r4-r11, lr}}
    sub     sp, sp, #64

    // t[0..15] is not zero-initialized. MUL_ROW_ZERO writes t[0..8] for
    // row 0 using movs-seeded UMAALs, and row i's final-carry STR writes
    // t[i+8] before any subsequent UMAAL reads it.

    // Preload b[0..7] into r4..r11. Stays live for all of Phase 1.
    ldm     r1, {{r4-r11}}

    // ----- Phase 1: schoolbook a * b -> t[0..16] -----
    MUL_ROW_ZERO
    MUL_ROW_REG 1
    MUL_ROW_REG 2
    MUL_ROW_REG 3
    MUL_ROW_REG 4
    MUL_ROW_REG 5
    MUL_ROW_REG 6
    MUL_ROW_REG 7

    // Reload r4..r11 with p[0..7] for Phase 2. b[] is no longer needed.
    ldm     r2, {{r4-r11}}

    // ----- Phase 2: Montgomery reduction, 8 rows with carry propagation -----
    REDUCE_INNER_REG 0
    ADD_CARRY_AT 32
    ADC_ZERO_AT 36
    ADC_ZERO_AT 40
    ADC_ZERO_AT 44
    ADC_ZERO_AT 48
    ADC_ZERO_AT 52
    ADC_ZERO_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 1
    ADD_CARRY_AT 36
    ADC_ZERO_AT 40
    ADC_ZERO_AT 44
    ADC_ZERO_AT 48
    ADC_ZERO_AT 52
    ADC_ZERO_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 2
    ADD_CARRY_AT 40
    ADC_ZERO_AT 44
    ADC_ZERO_AT 48
    ADC_ZERO_AT 52
    ADC_ZERO_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 3
    ADD_CARRY_AT 44
    ADC_ZERO_AT 48
    ADC_ZERO_AT 52
    ADC_ZERO_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 4
    ADD_CARRY_AT 48
    ADC_ZERO_AT 52
    ADC_ZERO_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 5
    ADD_CARRY_AT 52
    ADC_ZERO_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 6
    ADD_CARRY_AT 56
    ADC_ZERO_AT 60

    REDUCE_INNER_REG 7
    ADD_CARRY_AT 60

    // ----- Copy result t[8..16] back to this[0..8] -----
    ldr     r1, [sp, #32]
    str     r1, [r0, #0]
    ldr     r1, [sp, #36]
    str     r1, [r0, #4]
    ldr     r1, [sp, #40]
    str     r1, [r0, #8]
    ldr     r1, [sp, #44]
    str     r1, [r0, #12]
    ldr     r1, [sp, #48]
    str     r1, [r0, #16]
    ldr     r1, [sp, #52]
    str     r1, [r0, #20]
    ldr     r1, [sp, #56]
    str     r1, [r0, #24]
    ldr     r1, [sp, #60]
    str     r1, [r0, #28]

    add     sp, sp, #64
    pop     {{r4-r11, lr}}
    bx      lr

    .size mul_reduce_armv8m, . - mul_reduce_armv8m
    "#,
);

// ---------------------------------------------------------------------------
// u32-limb SOS Montgomery reduction, used as a design reference for the
// ARMv8-M asm that replaces `mul_reduce_asm`. The inner step
//     t[i+j] = t[i+j] + a[i]*b[j] + carry
// maps 1-to-1 onto UMAAL, wich is the instruction LLVM refuses to emit on
// thumbv8m.main. Keeping a host-runnable Rust version of the same algorithm
// lets us cross-check the asm's numerical output against a golden reference
// without round-tripping to hardware.
// ---------------------------------------------------------------------------

#[cfg(test)]
fn mul_reduce_u32_ref(
    this: &mut [u128; 2],
    by: &[u128; 2],
    modulus: &[u128; 2],
    inv: u128,
) {
    let a = split128(this);
    let b = split128(by);
    let p = split128(modulus);
    let inv_low = inv as u32;

    let mut t = [0u32; 17];

    // Phase 1: schoolbook a * b -> t[0..16]
    for i in 0..8 {
        let mut carry: u32 = 0;
        for j in 0..8 {
            let sum = (a[i] as u64) * (b[j] as u64) + (t[i + j] as u64) + (carry as u64);
            t[i + j] = sum as u32;
            carry = (sum >> 32) as u32;
        }
        t[i + 8] = carry;
    }

    // Phase 2: Montgomery reduction, 8 iterations
    for i in 0..8 {
        let m = t[i].wrapping_mul(inv_low);
        let mut carry: u32 = 0;
        for j in 0..8 {
            let sum = (m as u64) * (p[j] as u64) + (t[i + j] as u64) + (carry as u64);
            t[i + j] = sum as u32;
            carry = (sum >> 32) as u32;
        }
        // Propagate carry into higher words. For the BN254 Fq/Fr moduli
        // (both < 2^254) the carry always absorbs before t[16]; the extra
        // word is kept for robustness against any future odd modulus.
        let mut k = i + 8;
        while carry != 0 && k < 17 {
            let sum = (t[k] as u64) + (carry as u64);
            t[k] = sum as u32;
            carry = (sum >> 32) as u32;
            k += 1;
        }
    }

    let result = [t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]];
    *this = combine128(&result);
}

#[cfg(test)]
fn split128(x: &[u128; 2]) -> [u32; 8] {
    [
        x[0] as u32,
        (x[0] >> 32) as u32,
        (x[0] >> 64) as u32,
        (x[0] >> 96) as u32,
        x[1] as u32,
        (x[1] >> 32) as u32,
        (x[1] >> 64) as u32,
        (x[1] >> 96) as u32,
    ]
}

#[cfg(test)]
fn combine128(r: &[u32; 8]) -> [u128; 2] {
    [
        (r[0] as u128)
            | ((r[1] as u128) << 32)
            | ((r[2] as u128) << 64)
            | ((r[3] as u128) << 96),
        (r[4] as u128)
            | ((r[5] as u128) << 32)
            | ((r[6] as u128) << 64)
            | ((r[7] as u128) << 96),
    ]
}

#[cfg(test)]
mod mul_reduce_ref_tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    // BN254 Fq and Fr parameters, matching the `field_impl!` invocations in
    // src/fields/fp.rs. The [u64; 4] limb form from that file packs into
    // [u128; 2] as (hi64 << 64) | lo64 per limb pair.
    const FQ_MODULUS: [u128; 2] = [
        0x97816a916871ca8d3c208c16d87cfd47,
        0x30644e72e131a029b85045b68181585d,
    ];
    const FQ_INV: u128 = 0x9ede7d651eca6ac987d20782e4866389;

    const FR_MODULUS: [u128; 2] = [
        0x2833e84879b9709143e1f593f0000001,
        0x30644e72e131a029b85045b68181585d,
    ];
    const FR_INV: u128 = 0x6586864b4c6911b3c2e1f593efffffff;

    fn random_under(rng: &mut StdRng, modulus: &[u128; 2]) -> [u128; 2] {
        loop {
            let v: [u128; 2] = [rng.gen(), rng.gen()];
            if v[1] < modulus[1] || (v[1] == modulus[1] && v[0] < modulus[0]) {
                return v;
            }
        }
    }

    fn rand_reduce_matches(seed: [u8; 32], modulus: &[u128; 2], inv: u128, iters: usize) {
        let mut rng = StdRng::from_seed(seed);
        for _ in 0..iters {
            let a = random_under(&mut rng, modulus);
            let b = random_under(&mut rng, modulus);

            let mut via_rust = a;
            super::mul_reduce_rust(&mut via_rust, &b, modulus, inv);

            let mut via_ref = a;
            super::mul_reduce_u32_ref(&mut via_ref, &b, modulus, inv);

            assert_eq!(via_rust, via_ref, "mismatch: a={:x?} b={:x?}", a, b);
        }
    }

    #[test]
    fn fq_matches_rust() {
        rand_reduce_matches([0x11u8; 32], &FQ_MODULUS, FQ_INV, 10_000);
    }

    #[test]
    fn fr_matches_rust() {
        rand_reduce_matches([0x22u8; 32], &FR_MODULUS, FR_INV, 10_000);
    }
}

#[test]
fn setting_bits() {
    let rng = &mut ::rand::thread_rng();
    let modulo = U256::from([0xffffffffffffffff; 4]);

    let a = U256::random(rng, &modulo);
    let mut e = U256::zero();
    for (i, b) in a.bits().enumerate() {
        assert!(e.set_bit(255 - i, b));
    }

    assert_eq!(a, e);
}

#[test]
fn from_slice() {
    let tst = U256::one();
    let mut s = [0u8; 32];
    s[31] = 1;

    let num =
        U256::from_slice(&s).expect("U256 should initialize ok from slice in `from_slice` test");
    assert_eq!(num, tst);
}

#[test]
fn to_big_endian() {
    let num = U256::one();
    let mut s = [0u8; 32];

    num.to_big_endian(&mut s)
        .expect("U256 should convert to bytes ok in `to_big_endian` test");
    assert_eq!(
        s,
        [
            0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
            0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8,
        ]
    );
}

#[test]
fn testing_divrem() {
    let rng = &mut ::rand::thread_rng();

    let modulo = U256::from([
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ]);

    for _ in 0..100 {
        let c0 = U256::random(rng, &modulo);
        let c1 = U256::random(rng, &modulo);

        let c1q_plus_c0 = U512::new(&c1, &c0, &modulo);

        let (new_c1, new_c0) = c1q_plus_c0.divrem(&modulo);

        assert!(c1 == new_c1.unwrap());
        assert!(c0 == new_c0);
    }

    {
        // Modulus should become 1*q + 0
        let a = U512::from([
            0x3c208c16d87cfd47,
            0x97816a916871ca8d,
            0xb85045b68181585d,
            0x30644e72e131a029,
            0,
            0,
            0,
            0,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert_eq!(c1.unwrap(), U256::one());
        assert_eq!(c0, U256::zero());
    }

    {
        // Modulus squared minus 1 should be (q-1) q + q-1
        let a = U512::from([
            0x3b5458a2275d69b0,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert_eq!(
            c1.unwrap(),
            U256::from([
                0x3c208c16d87cfd46,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
        assert_eq!(
            c0,
            U256::from([
                0x3c208c16d87cfd46,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
    }

    {
        // Modulus squared minus 2 should be (q-1) q + q-2
        let a = U512::from([
            0x3b5458a2275d69af,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);

        assert_eq!(
            c1.unwrap(),
            U256::from([
                0x3c208c16d87cfd46,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
        assert_eq!(
            c0,
            U256::from([
                0x3c208c16d87cfd45,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
    }

    {
        // Ridiculously large number should fail
        let a = U512::from([
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert!(c1.is_none());
        assert_eq!(
            c0,
            U256::from([
                0xf32cfc5b538afa88,
                0xb5e71911d44501fb,
                0x47ab1eff0a417ff6,
                0x06d89f71cab8351f
            ])
        );
    }

    {
        // Modulus squared should fail
        let a = U512::from([
            0x3b5458a2275d69b1,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert!(c1.is_none());
        assert_eq!(c0, U256::zero());
    }

    {
        // Modulus squared plus one should fail
        let a = U512::from([
            0x3b5458a2275d69b2,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert!(c1.is_none());
        assert_eq!(c0, U256::one());
    }

    {
        let modulo = U256::from([
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);

        // Fr modulus masked off is valid
        let a = U512::from([
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x07ffffffffffffff,
        ]);

        let (c1, c0) = a.divrem(&modulo);

        assert!(c1.unwrap() < modulo);
        assert!(c0 < modulo);
    }
}
