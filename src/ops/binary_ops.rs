/// Implement +, -, *, / operators for Tensor
/// +=, -= are not supported but do as ops::inplace_add, ops::inplace_sub
extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::mem;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use tensor::{RawTensor, Tensor};


#[inline(always)]
fn scalar_tensor_add(a: &NdArray, b: &NdArray) -> NdArray
{
    // comparing the rank of tensors doesn't solve the problem here.
    if a.len() > b.len() { a + b } else { b + a }
}

#[inline(always)]
fn scalar_tensor_mul(a: &NdArray, b: &NdArray) -> NdArray
{
    // comparing the rank of tensors doesn't solve the problem here.
    if a.len() > b.len() { a * b } else { b * a }
}


pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;

pub struct InplaceAddOp;
pub struct InplaceSubOp;

impl ops::Op for AddOp {
    fn name(&self) -> &str
    {
        "AddOp"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let a = xs[0];
        let b = xs[1];
        scalar_tensor_add(a, b)
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy.clone()), Some(gy.clone())]
    }
}


impl ops::Op for SubOp {
    fn name(&self) -> &str
    {
        "SubOp"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let a = xs[0];
        let b = xs[1];
        if a.shape() == &[1] {
            // a is scalar
            // unwrap is safe
            let a = NdArray::from_elem(b.shape(), *a.get(0).unwrap());
            a - b
        } else {
            a - b
        }
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy.clone()), Some(-1 * gy)]
    }
}

impl ops::Op for MulOp {
    fn name(&self) -> &str
    {
        "MulOp"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let a = xs[0];
        let b = xs[1];
        scalar_tensor_mul(a, b)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x0 = inputs[0];
        let x1 = inputs[1];
        vec![Some(gy * x1), Some(gy * x0)]
    }
}

impl ops::Op for DivOp {
    fn name(&self) -> &str
    {
        "DivOp"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0 = xs[0];
        let x1 = xs[1];
        if x0.shape() == &[1] {
            // a is scalar
            // unwrap is safe
            NdArray::from_elem(x1.shape(), *x0.get(0).unwrap()) / x1
        } else {
            x0 / x1
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let a: Tensor = gy / x1;
        let b: Tensor = -1 * x0 * ops::pow(x1, -2.) * gy;
        vec![Some(a), Some(b)]
    }
}

impl ops::Op for InplaceAddOp {
    fn inplace(&self) -> bool
    {
        true
    }

    fn name(&self) -> &str
    {
        "InplaceAddOp"
    }

    fn compute_inplace(&self, xs: &mut [&mut NdArray], _: bool)
    {
        // safe transmute
        let b: &&NdArray = unsafe { mem::transmute(&mut xs[1]) };
        let a = &mut xs[0];
        a.zip_mut_with(b, |x, &y| *x += y);
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy.clone()), Some(gy.clone())]
    }
}


impl ops::Op for InplaceSubOp {
    fn inplace(&self) -> bool
    {
        true
    }

    fn name(&self) -> &str
    {
        "InplaceSubOp"
    }

    fn compute_inplace(&self, xs: &mut [&mut NdArray], _: bool)
    {
        // safe transmute
        let b: &&NdArray = unsafe { mem::transmute(&mut xs[1]) };
        let a = &mut xs[0];
        a.zip_mut_with(b, |x, &y| *x -= y);
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy.clone()), Some(-1 * gy)]
    }
}


#[inline]
fn scalar_to_tensor(arg: f32) -> Tensor
{
    Tensor(Rc::new(RawTensor {
        op: Box::new(ops::scalar::Scalar { val: arg }),
        inputs: vec![],
        top_rank: 0,
    }))
}

#[inline]
fn f32_to_tensor(arg: f32) -> Tensor
{
    scalar_to_tensor(arg)
}

#[inline]
fn f64_to_tensor(arg: f64) -> Tensor
{
    scalar_to_tensor(arg as f32)
}

#[inline]
fn i32_to_tensor(arg: i32) -> Tensor
{
    scalar_to_tensor(arg as f32)
}

#[inline]
fn i64_to_tensor(arg: i64) -> Tensor
{
    scalar_to_tensor(arg as f32)
}

#[inline]
fn u32_to_tensor(arg: u32) -> Tensor
{
    scalar_to_tensor(arg as f32)
}

#[inline]
fn u64_to_tensor(arg: u64) -> Tensor
{
    scalar_to_tensor(arg as f32)
}

#[inline]
fn usize_to_tensor(arg: usize) -> Tensor
{
    scalar_to_tensor(arg as f32)
}

#[inline]
fn isize_to_tensor(arg: isize) -> Tensor
{
    scalar_to_tensor(arg as f32)
}


// -- std::ops::{Add, Sub, Mul, Div} implementations --


macro_rules! impl_elementwise_between_tensor_and_scalar {
    (
        $trt:ident,
        $func:ident,
        $op:ident,
        $scalar_type:ty,
        $conversion_fn:ident
    ) => {

        // scalar op Tensor
        impl $trt<Tensor> for $scalar_type {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&$conversion_fn(self), &rhs])
            }
        }

        // scalar op &Tensor
        impl<'a> $trt<&'a Tensor> for $scalar_type {
            type Output = Tensor;
            fn $func(self, rhs: &'a Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&$conversion_fn(self), rhs])
            }
        }

        // Tensor op scalar
        impl $trt<$scalar_type> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: $scalar_type) -> Self::Output
            {
                ops::apply_op($op, &[&self, &$conversion_fn(rhs)])
            }
        }

        // &Tensor op scalar
        impl<'a> $trt<$scalar_type> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: $scalar_type) -> Self::Output
            {
                ops::apply_op($op, &[self, &$conversion_fn(rhs)])
            }
        }
    }
}

macro_rules! impl_elementwise_between_two_tensors {
    (
        $trt:ident,
        $func:ident,
        $op:ident
    ) => {
        // Tensor op Tensor
        impl $trt for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&self, &rhs])
            }
        }

        // Tensor op &Tensor
        impl<'a> $trt<&'a Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&self, rhs])
            }
        }

        // &Tensor op Tensor
        impl<'a> $trt<Tensor> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::apply_op($op, &[self, &rhs])
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'a, 'b> $trt<&'a Tensor> for &'b Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Self::Output
            {
                ops::apply_op($op, &[self, rhs])
            }
        }
    };
}



impl_elementwise_between_two_tensors!(Add, add, AddOp);
impl_elementwise_between_two_tensors!(Sub, sub, SubOp);
impl_elementwise_between_two_tensors!(Mul, mul, MulOp);
impl_elementwise_between_two_tensors!(Div, div, DivOp);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, i32, i32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, i32, i32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, i32, i32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, i32, i32_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, i64, i64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, i64, i64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, i64, i64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, i64, i64_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, f32, f32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, f32, f32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, f32, f32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, f32, f32_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, f64, f64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, f64, f64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, f64, f64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, f64, f64_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, u32, u32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, u32, u32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, u32, u32_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, u32, u32_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, u64, u64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, u64, u64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, u64, u64_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, u64, u64_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, usize, usize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, usize, usize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, usize, usize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, usize, usize_to_tensor);

impl_elementwise_between_tensor_and_scalar!(Add, add, AddOp, isize, isize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Sub, sub, SubOp, isize, isize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Mul, mul, MulOp, isize, isize_to_tensor);
impl_elementwise_between_tensor_and_scalar!(Div, div, DivOp, isize, isize_to_tensor);
