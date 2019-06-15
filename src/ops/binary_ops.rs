use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::ops;
use crate::tensor::Tensor;
use crate::Float;
/// Implement +, -, *, / operators for Tensor
/// +=, -=, *=, /= are provided as methods of ops::inplace_*.
/// *=, /= don't propagate gradients.
use ndarray;
use std::mem;

pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;
pub struct InplaceAddOp;
pub struct InplaceSubOp;
pub struct InplaceMulOp;
pub struct InplaceDivOp;
pub struct PreprocessBinOpGrad;
pub struct PreprocessBinOpGradGrad;

impl<T: Float> op::Op<T> for PreprocessBinOpGrad {
    fn name(&self) -> &str {
        "PreprocessBinOpGrad"
    }

    // Computes x's gradient.
    // Involves reduction as necessary.
    // Inputs: [gy, target_shape]
    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let gy = &xs[0];
        let x_shape_ = crate::ndarray_ext::vec_as_shape(&xs[1]);
        let x_shape = x_shape_.as_slice();
        let gy_shape = gy.shape();

        let ret = if x_shape == gy_shape {
            // The case where forward path didn't cause broadcast.
            Ok(crate::ArrRepr::View(gy.clone()))
        } else {
            // Broadcast occurred. We need reduction of `gy`.
            // First, handle the case where x is scalar.
            let x_is_scalar = crate::ndarray_ext::is_scalar_shape(x_shape);
            let x_shape = if x_is_scalar {
                vec![1; gy_shape.len()]
            } else {
                x_shape.to_vec()
            };
            // Reduce each dim as necessary
            let mut folded: Option<NdArray<T>> = None;
            for (i, (x_axis, gy_axis)) in x_shape.iter().zip(gy_shape).enumerate() {
                if x_axis < gy_axis {
                    if *x_axis == 1 {
                        // `fold_axis` squashes the axis automatically.
                        let axis = ndarray::Axis(if x_is_scalar { 0 } else { i });
                        let ret = match folded {
                            Some(ref a) => {
                                a.fold_axis(axis.clone(), T::zero(), |a, b| a.clone() + b.clone())
                            }
                            None => {
                                gy.fold_axis(axis.clone(), T::zero(), |a, b| a.clone() + b.clone())
                            }
                        };
                        if x_is_scalar {
                            mem::swap(&mut folded, &mut Some(ret));
                        } else {
                            // Expands squashed axis.
                            mem::swap(
                                &mut folded,
                                &mut Some(crate::ndarray_ext::expand_dims(ret, i)),
                            );
                        }
                    } else {
                        panic!("Incorrect gradient shape");
                    }
                }
                // case of x_axis < gy_axis: unreachable
                // case of x_axis == gy_axis: nothing to do
            }
            // TODO
            Ok(crate::ArrRepr::Owned(folded.unwrap()))
        };
        vec![ret]
    }

    // Do broadcast
    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x_shape = inputs[1];
        let gx = Tensor::builder()
            .set_inputs(vec![gy, x_shape])
            .build(PreprocessBinOpGradGrad);
        vec![Some(gx), None]
    }
}

// Do broadcast if necessary.
// Inputs: [gy, target_shape]
impl<T: Float> op::Op<T> for PreprocessBinOpGradGrad {
    fn name(&self) -> &str {
        "PreprocessBinOpGradGrad"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let gy = xs[0].clone();
        let target_shape_ = &xs[1];
        let target_shape_ = crate::ndarray_ext::vec_as_shape(target_shape_);
        let target_shape = target_shape_.as_slice();

        if gy.shape() == target_shape {
            return vec![Ok(crate::ArrRepr::View(gy))];
        }

        let gy_is_scalar = crate::ndarray_ext::is_scalar_shape(gy.shape());

        let mut gy = gy;

        // make broadcast dims if needed
        if gy_is_scalar {
            for &axis in target_shape.iter() {
                gy = crate::ndarray_ext::expand_dims_view(gy, axis);
            }
        }

        // do broadcast
        if let Some(ret) = gy.broadcast(target_shape) {
            vec![Ok(crate::ArrRepr::Owned(ret.to_owned()))]
        } else {
            panic!("Cant't broadcast.");
        }
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .build(PreprocessBinOpGrad);
        vec![Some(gx), None]
    }
}

impl<T: Float> op::Op<T> for AddOp {
    fn name(&self) -> &str {
        "Add"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(add_forward(&xs[0], &xs[1]))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let (gy1, gy2) = preprocess_gy(inputs[0], inputs[1], gy);
        vec![Some(gy1), Some(gy2)]
    }
}

impl<T: Float> op::Op<T> for SubOp {
    fn name(&self) -> &str {
        "Sub"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x0 = &xs[0];
        let x1 = &xs[1];
        let shape0: &[usize] = x0.shape();
        let ret = if shape0 == &[] {
            // is scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            crate::ArrRepr::Owned(x1.map(move |&a| x0_elem - a))
        } else {
            crate::ArrRepr::Owned(x0 - x1)
        };
        vec![Ok(ret)]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let (gy1, gy2) = preprocess_gy(inputs[0], inputs[1], gy);
        vec![Some(gy1), Some(ops::neg(&gy2))]
    }
}

impl<T: Float> op::Op<T> for MulOp {
    fn name(&self) -> &str {
        "Mul"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        vec![Ok(mul_forward(&xs[0], &xs[1]))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let (gy1, gy2) = preprocess_gy(x0, x1, gy);
        vec![Some(gy1 * x1), Some(gy2 * x0)]
    }
}

impl<T: Float> op::Op<T> for DivOp {
    fn name(&self) -> &str {
        "Div"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x0 = &xs[0];
        let x1 = &xs[1];
        let shape0: &[usize] = x0.shape();
        let shape1: &[usize] = x1.shape();
        let is_scalar0 = shape0 == &[] || shape0 == &[0];
        let is_scalar1 = shape1 == &[] || shape1 == &[1];
        let ret = if is_scalar0 {
            // a is a scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            x1.map(move |&a| x0_elem / a)
        } else if is_scalar1 {
            // b is a scalar
            let x1_elem = x1[ndarray::IxDyn(&[])];
            let rhs = T::one() / x1_elem;
            x0.mapv(|x0_elem| x0_elem * rhs)
        } else {
            x0 / x1
        };
        vec![Ok(crate::ArrRepr::Owned(ret))]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let (gy1, gy2) = preprocess_gy(x0, x1, gy);
        vec![
            Some(gy1 / x1),
            Some(ops::neg(x0) * ops::pow(x1, T::from(-2.).unwrap()) * gy2),
        ]
    }
}

// Reduce gy if broadcast occurred in the forward path.
fn preprocess_gy<T: Float>(
    x0: &Tensor<T>,
    x1: &Tensor<T>,
    gy: &Tensor<T>,
) -> (Tensor<T>, Tensor<T>) {
    let shape0 = x0.shape();
    let shape1 = x1.shape();
    let gy0 = Tensor::builder()
        .set_inputs(vec![gy, &shape0])
        .set_shape(shape0)
        .build(PreprocessBinOpGrad);
    let gy1 = Tensor::builder()
        .set_inputs(vec![gy, &shape1])
        .set_shape(shape1)
        .build(PreprocessBinOpGrad);
    (gy0, gy1)
}

macro_rules! impl_bin_op_forward {
    ($forward_name:ident, $bin_op:tt) => {
        fn $forward_name<'v, T: Float>(x0: &NdArrayView<'v, T>, x1: &NdArrayView<'v, T>) -> crate::ArrRepr<'v, T>
        {
            let shape0: &[usize]  = x0.shape();
            let shape1: &[usize]  = x1.shape();
            let scalar_shape = &[];
            let scalar_shape1 = &[0];

            let x0_is_scalar = shape0 == scalar_shape || shape0 == scalar_shape1;
            let x1_is_scalar = shape1 == scalar_shape || shape1 == scalar_shape1;

            let ret = if x0_is_scalar && !x1_is_scalar {
                let elem = x0[ndarray::IxDyn(&[])];
                x1.map(move |&a| a $bin_op elem)
            } else if x1_is_scalar && !x0_is_scalar {
                let elem = x1[ndarray::IxDyn(&[])];
                x0.map(move |&a| a $bin_op elem )
            } else if !x0_is_scalar && !x1_is_scalar {
                let len0: usize = shape0.iter().product();
                let len1: usize = shape1.iter().product();
                if len0 > len1 {
                    x0 $bin_op x1
                } else {
                    x1 $bin_op x0
                }
            } else {
                x0 $bin_op x1
            };
            crate::ArrRepr::Owned(ret)
        }
    };
}

impl_bin_op_forward!(add_forward, +);
impl_bin_op_forward!(mul_forward, *);
