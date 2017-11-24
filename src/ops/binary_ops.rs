/// Implement +, -, *, / operators for Tensor
/// +=, -= are provided as methods of ops::inplace_add, ops::inplace_sub
extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::mem;
use std::ops::{Add, Div, Mul, Sub};
use std::result::Result;
use tensor::Tensor;


pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;
pub struct InplaceAddOp;
pub struct InplaceSubOp;


impl ops::Op for AddOp {
    fn name(&self) -> &str
    {
        "Add"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        add_forward(xs[0], xs[1])
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let (gy1, gy2) = maybe_reduce_gy(inputs[0], inputs[1], gy);
        vec![Some(gy1), Some(gy2)]
    }
}


impl ops::Op for SubOp {
    fn name(&self) -> &str
    {
        "Sub"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let shape0: &[usize] = x0.shape();
        if shape0 == &[] {
            // a is scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            Ok(x1.map(move |a| x0_elem - a))
        } else {
            Ok(x0 - x1)
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let (gy1, gy2) = maybe_reduce_gy(inputs[0], inputs[1], gy);
        vec![Some(gy1), Some(gy2 * -1.)]
    }
}

impl ops::Op for MulOp {
    fn name(&self) -> &str
    {
        "Mul"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        mul_forward(xs[0], xs[1])
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let (gy1, gy2) = maybe_reduce_gy(x0, x1, gy);
        vec![Some(gy1 * x1), Some(gy2 * x0)]
    }
}

impl ops::Op for DivOp {
    fn name(&self) -> &str
    {
        "Div"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let shape0: &[usize] = x0.shape();
        if shape0 == &[] {
            // a is scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            Ok(x1.map(move |a| x0_elem / a))
        } else {
            Ok(x0 / x1)
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let (gy1, gy2) = maybe_reduce_gy(x0, x1, gy);
        vec![Some(gy1 / x1), Some(-1 * x0 * ops::pow(x1, -2.) * gy2)]
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

    fn compute_inplace(&self, xs: &mut [&mut NdArray]) -> Result<(), ::OpComputeErrorStatus>
    {
        // safe transmute probably
        let x1: &&NdArray = unsafe { mem::transmute(&mut xs[1]) };
        let x0 = &mut xs[0];
        x0.zip_mut_with(x1, |a, &b| *a += b);
        Ok(())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let (gy1, gy2) = maybe_reduce_gy(inputs[0], inputs[1], gy);
        vec![Some(gy1), Some(gy2)]
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

    fn compute_inplace(&self, xs: &mut [&mut NdArray]) -> Result<(), ::OpComputeErrorStatus>
    {
        // safe transmute probably
        let x1: &&NdArray = unsafe { mem::transmute(&mut xs[1]) };
        let x0 = &mut xs[0];
        x0.zip_mut_with(x1, |a, &b| *a -= b);
        Ok(())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let (gy1, gy2) = maybe_reduce_gy(inputs[0], inputs[1], gy);
        vec![Some(gy1), Some(-1. * gy2)]
    }
}

#[inline]
// Reduce gy if broadcast occurred in the forward pass
fn maybe_reduce_gy(x0: &Tensor, x1: &Tensor, gy: &Tensor) -> (Tensor, Tensor)
{
    let shape0 = x0.shape();
    let shape1 = x1.shape();
    let gy_shape = gy.shape();
    let sum0 = ops::reduction_ops::ReduceSum { keep_dims: true, sparse_axes: true };
    let sum1 = ops::reduction_ops::ReduceSum { keep_dims: true, sparse_axes: true };
    return (
        ops::apply_op(
            sum0,
            &[gy, &ops::not_equal(&gy_shape, &shape0)],
            Some(shape0),
        ), // gy1
        ops::apply_op(
            sum1,
            &[gy, &ops::not_equal(&gy_shape, &shape1)],
            Some(shape1),
        ), /* gy2 */
    );
}



// -- std::ops::{Add, Sub, Mul, Div} implementations --


macro_rules! impl_bin_op_between_tensor_and_scalar {
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
                ops::apply_op($op, &[&ops::scalar(self as f32), &rhs], Some(rhs.shape()))
            }
        }

        // scalar op &Tensor
        impl<'a> $trt<&'a Tensor> for $scalar_type {
            type Output = Tensor;
            fn $func(self, rhs: &'a Tensor) -> Self::Output
            {
                ops::apply_op($op, &[&ops::scalar(self as f32), rhs], Some(rhs.shape()))
            }
        }

        // Tensor op scalar
        impl $trt<$scalar_type> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: $scalar_type) -> Self::Output
            {
                ops::apply_op($op, &[&self, &ops::scalar(rhs as f32)], Some(self.shape()))
            }
        }

        // &Tensor op scalar
        impl<'a> $trt<$scalar_type> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: $scalar_type) -> Self::Output
            {
                ops::apply_op($op, &[self, &ops::scalar(rhs as f32)], Some(self.shape()))
            }
        }
    }
}

macro_rules! impl_bin_op_between_tensors {
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
                ops::$func(&self, &rhs)
            }
        }

        // Tensor op &Tensor
        impl<'a> $trt<&'a Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Self::Output
            {
                ops::$func(&self, rhs)
            }
        }

        // &Tensor op Tensor
        impl<'a> $trt<Tensor> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Self::Output
            {
                ops::$func(&self, &rhs)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'a, 'b> $trt<&'a Tensor> for &'b Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Self::Output
            {
                ops::$func(self, rhs)
            }
        }
    };
}

macro_rules! impl_bin_op_forward {
    ($forward_name:ident, $bin_op:tt) => {
        fn $forward_name(x0: &NdArray, x1: &NdArray) -> Result<NdArray, ::OpComputeErrorStatus>
        {
            let shape0: &[usize]  = x0.shape();
            let shape1: &[usize]  = x1.shape();
            let scalar_shape = &[];

            let x0_is_scalar = shape0 == scalar_shape;
            let x1_is_scalar = shape1 == scalar_shape;

            if x0_is_scalar && !x1_is_scalar {
                let elem = x0[ndarray::IxDyn(&[])];
                Ok(x1.map(move |a| a $bin_op elem ))
            } else if x1_is_scalar && !x0_is_scalar {
                let elem = x1[ndarray::IxDyn(&[])];
                Ok(x0.map(move |a| a $bin_op elem ))
            } else if !x0_is_scalar && !x1_is_scalar {
                let len0: usize = shape0.iter().product();
                let len1: usize = shape1.iter().product();
                if len0 > len1 {
                    Ok(x0 $bin_op x1)
                } else {
                    Ok(x1 $bin_op x0)
                }
            } else {
                Ok(x0 $bin_op x1)
            }
        }
    };
}

impl_bin_op_forward!(add_forward, +);
impl_bin_op_forward!(mul_forward, *);

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, i32, i32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, i32, i32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, i32, i32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, i32, i32_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, i64, i64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, i64, i64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, i64, i64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, i64, i64_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, f32, f32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, f32, f32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, f32, f32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, f32, f32_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, f64, f64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, f64, f64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, f64, f64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, f64, f64_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, u32, u32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, u32, u32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, u32, u32_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, u32, u32_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, u64, u64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, u64, u64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, u64, u64_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, u64, u64_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, usize, usize_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, usize, usize_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, usize, usize_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, usize, usize_to_tensor);

impl_bin_op_between_tensor_and_scalar!(Add, add, AddOp, isize, isize_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Sub, sub, SubOp, isize, isize_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Mul, mul, MulOp, isize, isize_to_tensor);
impl_bin_op_between_tensor_and_scalar!(Div, div, DivOp, isize, isize_to_tensor);
