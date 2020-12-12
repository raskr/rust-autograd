use crate::ndarray_ext;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::ops;
use crate::tensor::Tensor;
use crate::Float;
use crate::Graph;
use ndarray;
use std::f32;
use std::mem;

pub struct ReduceMin {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceMax {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceProd {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceSumToScalar;

pub struct ReduceSum {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceMean {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ArgMax {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ArgMin {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceGradCommon {
    pub should_make_broadcast_dims: bool,
    pub sparse_axes: bool,
}

macro_rules! impl_reduce_forward {
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:ident) => {
        fn $forward_name<'v, T: Float>(
            x: &NdArrayView<'v, T>,
            mut axes: Vec<usize>,
            keep_dims: bool,
        ) -> crate::ArrRepr<'v, T> {
            let x_shape = x.shape();

            if ndarray_ext::is_scalar_shape(x_shape) {
                // case of 0 rank
                crate::ArrRepr::View(x.clone())
            } else {
                // reduction axes are empty => do nothing
                if axes.is_empty() {
                    return crate::ArrRepr::View(x.clone());
                }

                // -- main logic --
                let mut folded: Option<NdArray<T>> = None;
                axes.sort();

                for axis in axes.into_iter().rev() {
                    let func = T::$reduce_fn_name;

                    let ret = match folded {
                        Some(ref a) => {
                            a.fold_axis(ndarray::Axis(axis), T::$reduce_default(), move |&l, &r| {
                                func(l, r)
                            })
                        }
                        None => {
                            x.fold_axis(ndarray::Axis(axis), T::$reduce_default(), move |&l, &r| {
                                func(l, r)
                            })
                        }
                    };

                    if keep_dims {
                        mem::swap(&mut folded, &mut Some(ndarray_ext::expand_dims(ret, axis)));
                    } else {
                        mem::swap(&mut folded, &mut Some(ret));
                    }
                }

                crate::ArrRepr::Owned(folded.unwrap_or_else(|| x.to_owned()))
            }
        }
    };
}

impl_reduce_forward!(compute_reduce_sum, add, zero);
impl_reduce_forward!(compute_reduce_min, min, max_value);
impl_reduce_forward!(compute_reduce_max, max, min_value);
impl_reduce_forward!(compute_reduce_prod, mul, one);

#[inline]
fn preprocess_axes<T: Float>(
    x: &NdArrayView<T>,
    axes: &NdArrayView<T>,
    sparse_axes: bool,
) -> Vec<usize> {
    if sparse_axes {
        ndarray_ext::sparse_to_dense(axes)
    } else {
        ndarray_ext::normalize_negative_axes(axes, x.ndim())
    }
}

impl<T: Float> op::Op<T> for ReduceSumToScalar {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        ctx.append_output(ndarray::arr0(x.sum()).into_dyn());
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder()
            .set_ro_inputs(&[&ctx.output_grad(), &ctx.graph().shape(ctx.input(0))])
            .build(ctx.graph(), ReduceSumToScalarGrad);
        ctx.append_input_grad(Some(gx))
    }
}

struct ReduceSumToScalarGrad;

impl<T: Float> op::Op<T> for ReduceSumToScalarGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = ndarray_ext::as_shape(&ctx.input(1));
        let ret = unsafe {
            let x = *ctx.input(0).as_ptr();
            ndarray::ArrayD::<T>::from_elem(ndarray::IxDyn(shape.as_slice()), x)
        };
        ctx.append_output(ret);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder()
            .append_input(&ctx.output_grad())
            .build(ctx.graph(), ReduceSumToScalar);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for ReduceSum {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        match compute_reduce_sum(x, axes, self.keep_dims) {
            crate::ArrRepr::Owned(ret) => ctx.append_output(ret),
            crate::ArrRepr::View(ret) => ctx.append_output_view(ret),
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder()
            .set_ro_inputs(&[
                &ctx.output_grad(),
                &ctx.graph().shape(&ctx.input(0)),
                &ctx.input(1),
            ])
            .build(ctx.graph(), grad_op);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for ReduceMean {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let x_shape = x.shape();
        if axes.is_empty() {
            return ctx.append_output_view(x.clone());
        }

        // Make reduction_len
        let mut reduction_len = 1.;
        for &axis in axes.iter() {
            reduction_len *= x_shape[axis as usize] as f32;
        }
        // Do summation
        let sum = compute_reduce_sum(x, axes, self.keep_dims);

        // Do division
        match sum {
            crate::ArrRepr::Owned(mut arr) => {
                let reduction_len_inv = T::one() / T::from(reduction_len).unwrap();
                arr.mapv_inplace(move |elem| elem * reduction_len_inv);
                ctx.append_output(arr)
            }
            crate::ArrRepr::View(view) => ctx.append_output_view(view),
        };
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let s = ctx.graph();
        let x = &ctx.input(0);
        let axes = &ctx.input(1);

        // Broadcast gy into x's shape
        let broadcast = Tensor::builder()
            .set_ro_inputs(&[&ctx.output_grad(), &s.shape(x), axes])
            .build(
                ctx.graph(),
                ReduceGradCommon {
                    should_make_broadcast_dims: !self.keep_dims,
                    sparse_axes: self.sparse_axes,
                },
            );

        // Divide
        let reduction_sizes = s.gather_common(s.shape(x), axes, 0);
        let reduction_len = s.reduce_prod(reduction_sizes, &[0], false);
        let gx = broadcast / reduction_len;

        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for ReduceProd {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        match compute_reduce_prod(x, axes, self.keep_dims) {
            crate::ArrRepr::Owned(ret) => {
                ctx.append_output(ret);
            }
            crate::ArrRepr::View(ret) => {
                ctx.append_output_view(ret);
            }
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let output = ctx.output();
        let tmp = Tensor::builder()
            .set_ro_inputs(&[&(gy * output), &ctx.graph().shape(x0), &x1])
            .build(ctx.graph(), grad_op);
        let gx = tmp / x0;
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for ReduceMin {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        match compute_reduce_min(x, axes, self.keep_dims) {
            crate::ArrRepr::Owned(ret) => {
                ctx.append_output(ret);
            }
            crate::ArrRepr::View(ret) => {
                ctx.append_output_view(ret);
            }
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        min_max_grad(
            &ctx.output_grad(),
            &ctx.input(0),
            &ctx.input(1),
            &ctx.output(),
            ctx.graph(),
            self.keep_dims,
            self.sparse_axes,
            ctx,
        );
    }
}

impl<T: Float> op::Op<T> for ReduceMax {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        match compute_reduce_max(x, axes, self.keep_dims) {
            crate::ArrRepr::Owned(ret) => {
                ctx.append_output(ret);
            }
            crate::ArrRepr::View(ret) => {
                ctx.append_output_view(ret);
            }
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        min_max_grad(
            &ctx.output_grad(),
            &ctx.input(0),
            &ctx.input(1),
            &ctx.output(),
            ctx.graph(),
            self.keep_dims,
            self.sparse_axes,
            ctx,
        );
    }
}

fn min_max_grad<'g, T: Float>(
    gy: &Tensor<'g, T>,
    x1: &Tensor<'g, T>,
    x2: &Tensor<'g, T>,
    y: &Tensor<'g, T>,
    s: &'g Graph<T>,
    keep_dims: bool,
    sparse_axes: bool,
    ctx: &mut op::GradientContext<'g, T>,
) {
    let grad_op1 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let grad_op2 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let x_shape = &s.shape(x1);
    let y = Tensor::builder()
        .set_ro_inputs(&[y, x_shape, x2])
        .build(s, grad_op1);
    let gy = Tensor::builder()
        .set_ro_inputs(&[gy, x_shape, x2])
        .build(s, grad_op2);
    let eq = s.equal(x1, y);
    ctx.append_input_grad(Some(s.mul(eq, gy)));
    ctx.append_input_grad(None);
}

fn argx_helper<T: Float>(
    x: &NdArrayView<T>,
    comp_fn: fn(T, T) -> T,
    default_val: T,
    keep_dim: bool,
    axis: isize,
) -> NdArray<T> {
    let axis = ndarray_ext::normalize_negative_axis(axis, x.ndim());
    let x_shape = x.shape();
    // 1. Make binary mask tensor (maximums are 1s)
    let mut mask = {
        let maxed = x.fold_axis(ndarray::Axis(axis), default_val, move |&a, &b| {
            comp_fn(a, b)
        });
        let mut mask = x.to_owned();
        let mut found = ndarray::Array::<bool, ndarray::IxDyn>::from_elem(maxed.shape(), false);
        for mut sub in mask.axis_iter_mut(ndarray::Axis(axis)) {
            ndarray::Zip::from(&mut sub)
                .and(&mut found)
                .and(&maxed)
                .apply(|r, f, m| {
                    let z = r == m && !*f;
                    *f = z;
                    *r = T::from(z as i32).unwrap();
                });
        }
        mask
    };

    // 2. Reshape the mask to 2-ranked. e.g. (2, 3, 4) -> (8, 3) (let `axis` be 1)
    let mask = {
        // move the `axis` to first, and put remaining together on the 2nd axis
        let reduction_len = x_shape[axis];
        ndarray_ext::roll_axis(&mut mask, ndarray::Axis(0), ndarray::Axis(axis));
        let shape2d = (reduction_len, mask.len() / reduction_len);
        let mut mask = mask.into_shape(shape2d).unwrap();
        mask.swap_axes(0, 1);
        mask
    };

    // 3. Make the indices (vertical vector)
    let indices = {
        let cols = mask.shape()[1];
        ndarray::Array::range(T::zero(), T::from(cols).unwrap(), T::one())
            .into_shape((cols, 1))
            .unwrap()
    };

    // 4. Dot product between mask and index-tensor
    let mat = mask.dot(&indices);

    // 5. Reshape it
    let mut final_shape = x_shape.to_vec();
    if keep_dim {
        final_shape[axis] = 1;
    } else {
        final_shape.remove(axis);
    }
    // unwrap is safe (95% confidence...)
    mat.into_dyn()
        .into_shape(ndarray::IxDyn(final_shape.as_slice()))
        .unwrap()
}

impl<T: Float> op::Op<T> for ArgMin {
    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let result = argx_helper(x, T::min, T::max_value(), self.keep_dim, self.axis);
        ctx.append_output(result);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None)
    }
}

impl<T: Float> op::Op<T> for ArgMax {
    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x = &ctx.input(0);
        let result = argx_helper(x, T::max, T::min_value(), self.keep_dim, self.axis);
        ctx.append_output(result);
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None)
    }
}

impl<T: Float> op::Op<T> for ReduceGradCommon {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        //  broadcast `gy` into `target_shape`
        let gy = ctx.input(0);
        let target_shape = ndarray_ext::as_shape(&ctx.input(1)); // x's shape

        if gy.shape() == target_shape.as_slice() {
            return ctx.append_output_view(gy.clone());
        }

        let x_is_scalar = ndarray_ext::is_scalar_shape(gy.shape());

        // make broadcast dims if needed
        if self.should_make_broadcast_dims || x_is_scalar {
            let axes = &ctx.input(2);

            // convert axes to usize vec
            let mut axes = if self.sparse_axes {
                ndarray_ext::sparse_to_dense(axes)
            } else {
                ndarray_ext::normalize_negative_axes(axes, target_shape.len())
            };

            let mut gy_shape = gy.shape().to_vec();
            axes.sort();
            for &axis in axes.iter() {
                gy_shape.insert(axis, 1);
            }
            // do broadcast
            let a = gy.into_shape(gy_shape).unwrap();
            ctx.append_output(a.broadcast(target_shape).unwrap().to_owned())
        } else {
            // do broadcast
            ctx.append_output(gy.broadcast(target_shape).unwrap().to_owned())
        }
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let sum = ops::reduction_ops::ReduceSum {
            keep_dims: self.should_make_broadcast_dims,
            sparse_axes: self.sparse_axes,
        };
        let axes = &ctx.input(2);
        let gx = Tensor::builder()
            .set_ro_inputs(&[&ctx.output_grad(), axes])
            .build(ctx.graph(), sum);
        ctx.append_input_grad(Some(gx));
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}
