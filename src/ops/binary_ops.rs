use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use crate::Graph;
/// Implement +, -, *, / operators for Tensor
/// +=, -=, *=, /= are provided as methods of c.inplace_*.
/// *=, /= don't propagate gradients.
use ndarray;
use std::mem;

pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;
pub struct PreprocessBinOpGrad;
pub struct PreprocessBinOpGradGrad;

#[cfg(feature = "mkl")]
macro_rules! bin_op_same_shape {
    ($vms_op:ident, $vmd_op:ident, $std_op:tt, $a:expr, $b:expr) => {
        unsafe {
            if same_type::<T, f32>() {
                let mut y = Vec::with_capacity($a.len());
                $vms_op($a.len() as MklInt, $a.as_ptr() as *const f32, $b.as_ptr() as *const f32, y.as_mut_ptr() as *mut f32);
                y.set_len($a.len());
                NdArray::from_shape_vec_unchecked($a.shape(), y)
            } else if same_type::<T, f64>() {
                let mut y = Vec::with_capacity($a.len());
                $vmd_op($a.len() as MklInt, $a.as_ptr() as *const f64, $b.as_ptr() as *const f64, y.as_mut_ptr() as *mut f64);
                y.set_len($a.len());
                NdArray::from_shape_vec_unchecked($a.shape(), y)
            } else {
                $a $std_op $b
            }
        }
    };
}

impl<T: Float> op::Op<T> for PreprocessBinOpGrad {
    // Computes x's gradient.
    // Involves reduction as necessary.
    // Inputs: [gy, target_shape]
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let gy = ctx.input(0);
        let x_shape_ = crate::ndarray_ext::as_shape(&ctx.input(1));
        let x_shape = x_shape_.as_slice();
        let gy_shape = gy.shape();

        if x_shape == gy_shape {
            // The case where forward path didn't cause broadcast.
            ctx.append_output_view(gy.clone());
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
                            Some(ref a) => a.fold_axis(axis.clone(), T::zero(), |&a, &b| a + b),
                            None => gy.fold_axis(axis.clone(), T::zero(), |&a, &b| a + b),
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
                        ctx.set_error(op::OpError::IncompatibleShape(
                            "Incorrect gradient shape".to_string(),
                        ));
                        return;
                    }
                }
                // case of x_axis < gy_axis: unreachable
                // case of x_axis == gy_axis: nothing to do
            }
            // TODO
            ctx.append_output(folded.unwrap());
        };
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder()
            .set_ro_inputs(&[&ctx.output_grad(), &ctx.input(1)])
            .build(ctx.graph(), PreprocessBinOpGradGrad);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

// Do broadcast if necessary.
// Inputs: [gy, target_shape]
impl<T: Float> op::Op<T> for PreprocessBinOpGradGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let target_shape_ = ctx.input(1);
        let target_shape_ = crate::ndarray_ext::as_shape(&target_shape_);
        let target_shape = target_shape_.as_slice();

        let gy = ctx.input(0);
        if gy.shape() == target_shape {
            ctx.append_output_view(gy);
            return;
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
            ctx.append_output(ret.to_owned());
        } else {
            ctx.set_error(op::OpError::IncompatibleShape(
                "PreprocessBinOpGradGrad: Cant't broadcast.".to_string(),
            ));
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder()
            .set_ro_inputs(&[&ctx.input(0), &ctx.output_grad()])
            .build(ctx.graph(), PreprocessBinOpGrad);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for AddOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = add_forward(&ctx.input(0), &ctx.input(1));
        ctx.append_output(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let shape0 = &ctx.graph().shape(x0);
        let shape1 = &ctx.graph().shape(x1);
        let gy0 = reduce_if_necessary(shape0, &gy, g);
        let gy1 = reduce_if_necessary(shape1, &gy, g);
        ctx.append_input_grad(Some(gy0));
        ctx.append_input_grad(Some(gy1));
    }
}

impl<T: Float> op::Op<T> for SubOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let shape0: &[usize] = x0.shape();
        let shape1: &[usize] = x1.shape();
        let ret = if shape0 == [] {
            // is scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            x1.map(move |&a| x0_elem - a)
        } else if shape0 == shape1 {
            #[cfg(feature = "mkl")]
            {
                use crate::{ops::mkl_ffi::*, same_type};
                bin_op_same_shape!(vsSub, vdSub, -, x0, x1)
            }
            #[cfg(not(feature = "mkl"))]
            {
                x0 - x1
            }
        } else {
            x0 - x1
        };
        ctx.append_output(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let shape0 = &ctx.graph().shape(x0);
        let shape1 = &ctx.graph().shape(x1);
        let (gy1, gy2) = preprocess_gy(shape0, shape1, &ctx.output_grad(), ctx.graph());
        ctx.append_input_grad(Some(gy1));
        ctx.append_input_grad(Some(ctx.graph().neg(&gy2)));
    }
}

impl<T: Float> op::Op<T> for MulOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = mul_forward(&ctx.input(0), &ctx.input(1));
        ctx.append_output(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let graph = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);

        let gy = ctx.output_grad();

        let gx0 = gy * x1;
        let gx1 = gy * x0;

        let shape0 = &graph.shape(x0);
        let shape1 = &graph.shape(x1);

        let gx0 = reduce_if_necessary(shape0, &gx0, graph);
        let gx1 = reduce_if_necessary(shape1, &gx1, graph);

        ctx.append_input_grad(Some(gx0));
        ctx.append_input_grad(Some(gx1));
    }
}

impl<T: Float> op::Op<T> for DivOp {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let shape0: &[usize] = x0.shape();
        let shape1: &[usize] = x1.shape();
        let is_scalar0 = shape0 == [] || shape0 == [0];
        let is_scalar1 = shape1 == [] || shape1 == [1];
        let ret = if is_scalar0 {
            // a is a scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            x1.map(move |&a| x0_elem / a)
        } else if is_scalar1 {
            // b is a scalar
            let x1_elem = x1[ndarray::IxDyn(&[])];
            let rhs = T::one() / x1_elem;
            x0.mapv(|x0_elem| x0_elem * rhs)
        } else if shape0 == shape1 {
            #[cfg(feature = "mkl")]
            {
                use crate::{ops::mkl_ffi::*, same_type};
                bin_op_same_shape!(vsDiv, vdDiv, /, x0, x1)
            }
            #[cfg(not(feature = "mkl"))]
            {
                x0 / x1
            }
        } else {
            x0 / x1
        };
        ctx.append_output(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let shape0 = &g.shape(x0);
        let shape1 = &g.shape(x1);
        let gy = ctx.output_grad();

        let gx0 = gy / x1;
        let gx1 = g.neg(x0) * g.pow(x1, T::from(-2.).unwrap()) * gy;

        let gx0 = reduce_if_necessary(shape0, &gx0, g);
        let gx1 = reduce_if_necessary(shape1, &gx1, g);

        ctx.append_input_grad(Some(gx0));
        ctx.append_input_grad(Some(gx1));
    }
}

fn reduce_if_necessary<'g, T: Float>(
    target_shape: &Tensor<'g, T>,
    x: &Tensor<'g, T>,
    graph: &'g Graph<T>,
) -> Tensor<'g, T> {
    Tensor::builder()
        .set_ro_inputs(&[x, target_shape])
        .set_shape(target_shape)
        .build(graph, PreprocessBinOpGrad)
}

// Reduce gy if broadcast occurred in the forward path.
fn preprocess_gy<'b, T: Float>(
    shape0: &Tensor<'b, T>,
    shape1: &Tensor<'b, T>,
    gy: &Tensor<'b, T>,
    c: &'b Graph<T>,
) -> (Tensor<'b, T>, Tensor<'b, T>) {
    let gy0 = Tensor::builder()
        .set_ro_inputs(&[gy, shape0])
        .set_shape(shape0)
        .build(c, PreprocessBinOpGrad);
    let gy1 = Tensor::builder()
        .set_ro_inputs(&[gy, shape1])
        .set_shape(shape1)
        .build(c, PreprocessBinOpGrad);
    (gy0, gy1)
}

macro_rules! impl_bin_op_forward {
    ($forward_name:ident, $bin_op:tt, $vms_op:ident, $vmd_op:ident) => {
        fn $forward_name<'v, T: Float>(x0: &NdArrayView<'v, T>, x1: &NdArrayView<'v, T>) -> NdArray<T>
        {
            let shape0: &[usize] = x0.shape();
            let shape1: &[usize] = x1.shape();
            let scalar_shape = &[];
            let scalar_shape1 = &[0];

            let x0_is_scalar = shape0 == scalar_shape || shape0 == scalar_shape1;
            let x1_is_scalar = shape1 == scalar_shape || shape1 == scalar_shape1;

            if x0_is_scalar && !x1_is_scalar {
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
                    // tensor vs tensor (same shapes)
                    #[cfg(feature = "mkl")]
                    {
                        use crate::{ops::mkl_ffi::*, same_type};
                        bin_op_same_shape!($vms_op, $vmd_op, $bin_op, x0, x1)
                    }
                    #[cfg(not(feature = "mkl"))] {
                        x0 $bin_op x1
                    }
                }
            } else {
                // scalar vs scalar
                x0 $bin_op x1
            }
        }
    };
}

impl_bin_op_forward!(add_forward, +, vsAdd, vdAdd);
impl_bin_op_forward!(mul_forward, *, vsMul, vdMul);
