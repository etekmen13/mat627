#![allow(dead_code)]

use std::marker::PhantomData;

/// Natural numbers as types, needed for symbolic differentiation at compile time.

pub struct Zero;
pub struct Succ<N>(PhantomData<N>);

pub trait Nat {
    const VALUE: usize;
    type Pred: Nat;
}

impl Nat for Zero {
    const VALUE: usize = 0;
    type Pred = Zero;
}
impl<N: Nat> Nat for Succ<N> {
    const VALUE: usize = 1 + N::VALUE;
    type Pred = N;
}

#[macro_export]
macro_rules! nat {
    (0) => { $crate::peano::Zero };
    (1) => { $crate::peano::Succ<$crate::peano::Zero> };
    (2) => { $crate::peano::Succ<$crate::peano::Succ<$crate::peano::Zero>> };
    (3) => { $crate::peano::Succ<$crate::peano::Succ<$crate::peano::Succ<$crate::peano::Zero>>> };
    (4) => { $crate::peano::Succ<$crate::peano::Succ<$crate::peano::Succ<$crate::peano::Succ<$crate::peano::Zero>>>> };
    (5) => { $crate::peano::Succ<$crate::peano::Succ<$crate::peano::Succ<$crate::peano::Succ<$crate::peano::Succ<$crate::peano::Zero>>>>> };
}
