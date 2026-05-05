#![allow(dead_code, unused_macros)]

use crate::peano::{Nat, Succ};
use std::marker::PhantomData;
/// Expression trait for symbolic differentiation.
/// Every expression must declare the expression of its derivative.
pub trait Expr: Copy {
    type Diff: Expr;
    fn eval(&self, x: f64) -> f64;
    fn diff(&self) -> Self::Diff;
}

#[derive(Clone, Copy)]
pub struct Const(pub f64);
#[derive(Clone, Copy)]
pub struct Var;
#[derive(Clone, Copy)]
pub struct Add<L, R>(pub L, pub R);
#[derive(Clone, Copy)]
pub struct Mul<L, R>(pub L, pub R);
pub struct Pow<T, N>(pub T, pub PhantomData<N>);
impl<T: Clone, N> Clone for Pow<T, N> {
    fn clone(&self) -> Self {
        Pow(self.0.clone(), PhantomData)
    }
}

impl<T: Copy, N> Copy for Pow<T, N> {}

#[derive(Clone, Copy)]
pub struct Exp<T>(pub T);

#[derive(Clone, Copy)]
pub struct Neg<T>(pub T);
#[derive(Clone, Copy)]
struct Div<L, R>(pub L, pub R);

impl<L: Expr, R: Expr> Expr for Div<L, R> {
    type Diff = Div<Add<Mul<L::Diff, R>, Neg<Mul<L, R::Diff>>>, Mul<R, R>>; // quotient rule

    fn eval(&self, x: f64) -> f64 {
        self.0.eval(x) / self.1.eval(x)
    }

    fn diff(&self) -> Self::Diff {
        Div(
            Add(Mul(self.0.diff(), self.1), Neg(Mul(self.0, self.1.diff()))),
            Mul(self.1, self.1),
        )
    }
}
impl Expr for Const {
    type Diff = Const;
    fn eval(&self, _: f64) -> f64 {
        self.0
    }
    fn diff(&self) -> Const {
        Const(0.0)
    }
}
impl<T: Expr> Expr for Neg<T> {
    type Diff = Neg<T::Diff>;
    fn eval(&self, x: f64) -> f64 {
        -self.0.eval(x)
    }
    fn diff(&self) -> Self::Diff {
        Neg(self.0.diff())
    }
}

impl Expr for Var {
    type Diff = Const;
    fn eval(&self, x: f64) -> f64 {
        x
    }
    fn diff(&self) -> Const {
        Const(1.0)
    }
}

/// (f+g)' = f' + g'
impl<L: Expr, R: Expr> Expr for Add<L, R> {
    type Diff = Add<L::Diff, R::Diff>;
    fn eval(&self, x: f64) -> f64 {
        self.0.eval(x) + self.1.eval(x)
    }
    fn diff(&self) -> Self::Diff {
        Add(self.0.diff(), self.1.diff())
    }
}
/// (fg)' = f'g + fg'
impl<L: Expr, R: Expr> Expr for Mul<L, R> {
    type Diff = Add<Mul<L::Diff, R>, Mul<L, R::Diff>>;
    fn eval(&self, x: f64) -> f64 {
        self.0.eval(x) * self.1.eval(x)
    }
    fn diff(&self) -> Self::Diff {
        Add(Mul(self.0.diff(), self.1), Mul(self.0, self.1.diff()))
    }
}

impl<T: Expr> Expr for Pow<T, nat!(0)> {
    type Diff = Const;
    fn eval(&self, _: f64) -> f64 {
        1.0
    }
    fn diff(&self) -> Const {
        Const(0.0)
    }
}

impl<T: Expr, N: Nat> Expr for Pow<T, Succ<N>>
where
    Pow<T, N>: Expr,
{
    type Diff = Mul<Mul<Const, Pow<T, N>>, T::Diff>;
    fn eval(&self, x: f64) -> f64 {
        self.0.eval(x).powi(Succ::<N>::VALUE as i32)
    }
    fn diff(&self) -> Self::Diff {
        Mul(
            Mul(Const(Succ::<N>::VALUE as f64), Pow(self.0, PhantomData)),
            self.0.diff(),
        )
    }
}

impl<T: Expr> Expr for Exp<T> {
    type Diff = Mul<T::Diff, Exp<T>>;
    fn eval(&self, x: f64) -> f64 {
        self.0.eval(x).exp()
    }
    fn diff(&self) -> Self::Diff {
        Mul(self.0.diff(), Exp(self.0))
    }
}
pub(crate) trait NthDiff<N>: Expr {
    type Output: Expr;

    fn nth_diff(self) -> Self::Output;
}

impl<E: Expr> NthDiff<nat!(0)> for E {
    type Output = E;

    fn nth_diff(self) -> Self::Output {
        self
    }
}

impl<E, N> NthDiff<Succ<N>> for E
where
    E: Expr,
    N: Nat,
    E::Diff: NthDiff<N>,
{
    type Output = <E::Diff as NthDiff<N>>::Output;

    fn nth_diff(self) -> Self::Output {
        <E::Diff as NthDiff<N>>::nth_diff(self.diff())
    }
}

pub(crate) fn nth_diff<N, E>(f: E) -> <E as NthDiff<N>>::Output
where
    E: NthDiff<N>,
{
    <E as NthDiff<N>>::nth_diff(f)
}

macro_rules! expr_add {
    (@acc [$($acc:tt)*] @) => {
        expr_mul!($($acc)*)
    };
    (@acc [$($acc:tt)*] @ + $($rest:tt)*) => {
        $crate::diff::Add(expr_mul!($($acc)*), expr!($($rest)*))
    };
    (@acc [$($acc:tt)*] @ - $($rest:tt)*) => {
        $crate::diff::Add(expr_mul!($($acc)*), $crate::diff::Neg(expr!($($rest)*)))
    };
    (@acc [$($acc:tt)*] @ $t:tt $($rest:tt)*) => {
        expr_add!(@acc [$($acc)* $t] @ $($rest)*)
    };
    ($($t:tt)*) => {
        expr_add!(@acc [] @ $($t)*)
    };
}

macro_rules! expr_mul {
    (@acc [$($acc:tt)*] @) => {
        expr_pow!($($acc)*)
    };
    (@acc [$($acc:tt)*] @ * $($rest:tt)*) => {
        $crate::diff::Mul(expr_pow!($($acc)*), expr_mul!($($rest)*))
    };
    (@acc [$($acc:tt)*] @ / $($rest:tt)*) => {
        $crate::diff::Div(expr_pow!($($acc)*), expr_mul!($($rest)*))
    };
    (@acc [$($acc:tt)*] @ $t:tt $($rest:tt)*) => {
        expr_mul!(@acc [$($acc)* $t] @ $($rest)*)
    };
    ($($t:tt)*) => {
        expr_mul!(@acc [] @ $($t)*)
    };
}

macro_rules! expr_pow {
    (@acc [$($acc:tt)*] @) => {
        expr_atom!($($acc)*)
    };

    (@acc [$($acc:tt)*] @ ^ $n:tt $($rest:tt)*) => {
        $crate::diff::Pow(
            expr_atom!($($acc)*),
            std::marker::PhantomData::<$crate::nat!($n)>
        )
    };

    (@acc [$($acc:tt)*] @ $t:tt $($rest:tt)*) => {
        expr_pow!(@acc [$($acc)* $t] @ $($rest)*)
    };

    ($($t:tt)*) => {
        expr_pow!(@acc [] @ $($t)*)
    };
}

macro_rules! expr_atom {
    (x) => {
        $crate::diff::Var
    };
    ($n:literal) => {
        $crate::diff::Const($n as f64)
    };
    (($($inner:tt)*)) => {
        expr!($($inner)*)
    };
    (exp ($($inner:tt)*)) => {
        $crate::diff::Exp(expr!($($inner)*))
    };
}

macro_rules! expr {
    ($($t:tt)*) => { expr_add!(@acc [] @ $($t)*) };
}
