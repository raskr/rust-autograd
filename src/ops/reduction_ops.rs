use ndarray;
use ndarray_ext;
use ndarray_ext::NdArray;
use op;
use ops;
use std::f32;
use std::mem;
use std::ops::Add;
use std::ops::Mul;
use tensor::Tensor;

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

pub struct ReduceGradCommon {
    pub should_make_broadcast_dims: bool,
    pub sparse_axes: bool,
}

macro_rules! impl_reduce_forward {
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:expr) => {
        fn $forward_name(
            x: &NdArray,
            mut axes: Vec<usize>,
            keep_dims: bool,
        ) -> Result<NdArray, ::op::ComputeError> {
            let x_shape = x.shape();

            if ndarray_ext::is_scalar_shape(x_shape) {
                // case of 0 rank
                Ok((*x).clone())
            } else {
                // reduction axes are empty => do nothing
                if axes.is_empty() {
                    return Err(::op::ComputeError::Delegate { to: 0 });
                }

                // -- main logic --
                let mut folded: Option<NdArray> = None;
                axes.sort();

                for axis in axes.into_iter().rev() {
                    let func = f32::$reduce_fn_name;
                    let ret = folded.as_ref().unwrap_or(x).fold_axis(
                        ndarray::Axis(axis),
                        $reduce_default,
                        move |&a, &b| func(a, b),
                    );

                    if keep_dims {
                        mem::swap(&mut folded, &mut Some(ndarray_ext::expand_dims(ret, axis)));
                    } else {
                        mem::swap(&mut folded, &mut Some(ret));
                    }
                }

                Ok(folded.unwrap_or_else(|| x.clone()))
            }
        }
    };
}

impl_reduce_forward!(compute_reduce_sum, add, 0.);
impl_reduce_forward!(compute_reduce_min, min, f32::MAX);
impl_reduce_forward!(compute_reduce_max, max, f32::MIN);
impl_reduce_forward!(compute_reduce_prod, mul, 1.);

#[inline]
fn preprocess_axes(x: &NdArray, axes: &NdArray, sparse_axes: bool) -> Vec<usize> {
    if sparse_axes {
        ndarray_ext::sparse_to_dense(axes)
    } else {
        ndarray_ext::normalize_negative_axes(axes, x.ndim())
    }
}

impl op::Op for ReduceSum {
    fn name(&self) -> &str {
        "ReduceSum"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        vec![compute_reduce_sum(x, axes, self.keep_dims)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder()
            .set_inputs(vec![gy, &inputs[0].shape(), inputs[1]])
            .build(grad_op);
        vec![Some(gx), None]
    }
}

impl op::Op for ReduceMean {
    fn name(&self) -> &str {
        "ReduceMean"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        let x_shape = x.shape();
        if axes.is_empty() {
            return vec![Err(::op::ComputeError::Delegate { to: 0 })];
        }

        // Make reduction_len
        let mut reduction_len = 1.;
        for &axis in axes.iter() {
            reduction_len *= x_shape[axis as usize] as f32;
        }

        // Do summation
        let sum: Result<NdArray, _> = compute_reduce_sum(x, axes, self.keep_dims);

        // Do division
        let ret = sum.map(|mut ok| {
            ok *= 1. / reduction_len;
            ok
        });

        vec![ret]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let x = inputs[0];
        let axes = inputs[1];

        // Broadcast gy into x's shape
        let broadcast = Tensor::builder()
            .set_inputs(vec![gy, &inputs[0].shape(), inputs[1]])
            .build(ReduceGradCommon {
                should_make_broadcast_dims: !self.keep_dims,
                sparse_axes: self.sparse_axes,
            });

        // Divide
        let reduction_sizes = &ops::gather_common(&x.shape(), axes, 0);
        let reduction_len = &ops::reduce_prod(reduction_sizes, &[0], false); // 1: &[]
        let reciprocal = ops::reciprocal(reduction_len); // 1: &[]
        let gx = broadcast * reciprocal;

        vec![Some(gx), None]
    }
}

impl op::Op for ReduceProd {
    fn name(&self) -> &str {
        "ReduceProd"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        let ret = compute_reduce_prod(x, axes, self.keep_dims);
        vec![ret]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let tmp = Tensor::builder()
            .set_inputs(vec![&(gy * output), &inputs[0].shape(), inputs[1]])
            .build(grad_op);
        let gx = tmp / inputs[0];
        vec![Some(gx), None]
    }
}

impl op::Op for ReduceMin {
    fn name(&self) -> &str {
        "ReduceMin"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        vec![compute_reduce_min(x, axes, self.keep_dims)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

impl op::Op for ReduceMax {
    fn name(&self) -> &str {
        "ReduceMax"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        vec![compute_reduce_max(x, axes, self.keep_dims)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

fn min_max_grad(
    gy: &Tensor,
    inputs: &[&Tensor],
    output: &Tensor,
    keep_dims: bool,
    sparse_axes: bool,
) -> Vec<Option<Tensor>> {
    let grad_op1 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let grad_op2 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let x = inputs[0];
    let x_shape = inputs[0].shape();
    let y = Tensor::builder()
        .set_inputs(vec![output, &x_shape, inputs[1]])
        .build(grad_op1);
    let gy = Tensor::builder()
        .set_inputs(vec![gy, &x_shape, inputs[1]])
        .build(grad_op2);
    let eq = ops::equal(&x, &y);
    vec![Some(ops::mul_inplace(eq, &gy)), None]
}

impl op::Op for ArgMax {
    fn name(&self) -> &str {
        "ArgMax"
    }

    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let x_shape = x.shape();

        // 1. Make binary mask tensor (maximum is 1)
        let mut mask = {
            let max_fn = f32::max;
            let maxed = x.fold_axis(ndarray::Axis(axis), f32::MIN, move |&a, &b| max_fn(a, b));
            let mut mask = x.clone();
            let mut found = ndarray::Array::<bool, ndarray::IxDyn>::from_elem(maxed.shape(), false);
            for mut sub in mask.axis_iter_mut(ndarray::Axis(axis)) {
                ndarray::Zip::from(&mut sub)
                    .and(&mut found)
                    .and(&maxed)
                    .apply(|r, f, m| {
                        let z = r == m && !*f;
                        *f = z;
                        *r = (z as i32) as f32;
                    });
            }
            mask
        };

        // 2. Reshape the mask to 2-ranked. e.g. (2, 3, 4) -> (8, 3) (let `axis` be 1)
        let mask = {
            // move axis to first, and remaining is put together in the 2nd axis
            let reduction_len = x_shape[axis];
            ndarray_ext::roll_axis(&mut mask, ndarray::Axis(0), ndarray::Axis(axis));
            let shape2d = (reduction_len, mask.len() / reduction_len);
            // unwrap is safe
            let mut mask = mask.into_shape(shape2d).unwrap();
            mask.swap_axes(0, 1);
            mask
        };

        // 3. Make indices (vertical vector)
        let indices = {
            let cols = mask.shape()[1];
            // unwrap is safe
            ndarray::Array::range(0., cols as f32, 1.)
                .into_shape((cols, 1))
                .unwrap()
        };

        // 4. Dot product between mask and index-tensor
        let mat = mask.dot(&indices);

        // 5. reshape it
        let result = {
            let mut final_shape = x_shape.to_vec();
            if self.keep_dim {
                final_shape[axis] = 1;
            } else {
                final_shape.remove(axis);
            }
            // unwrap is safe (95% confidence...)
            mat.into_dyn()
                .into_shape(ndarray::IxDyn(final_shape.as_slice()))
                .unwrap()
        };

        vec![Ok(result)]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}

impl op::Op for ReduceGradCommon {
    fn name(&self) -> &str {
        "ReduceGradCommon"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult {
        let xs = ctx.grab_inputs();
        //  broadcast `gy` into `target_shape`
        let gy = xs[0];
        let target_shape = ndarray_ext::vec_as_shape(xs[1]); // x's shape

        if gy.shape() == target_shape.as_slice() {
            return vec![Err(::op::ComputeError::Delegate { to: 0 })];
        }

        let x_is_scalar = ndarray_ext::is_scalar_shape(gy.shape());

        let ret = {
            let mut gy_view = gy.view();

            // make broadcast dims if needed
            if self.should_make_broadcast_dims || x_is_scalar {
                let axes = xs[2];

                // convert axes to usize vec
                let mut axes = if self.sparse_axes {
                    ndarray_ext::sparse_to_dense(axes)
                } else {
                    ndarray_ext::normalize_negative_axes(axes, target_shape.len())
                };

                let mut gy_shape = gy.shape().to_vec();
                axes.sort();
                for &axis in axes.iter() {
                    assert!(
                        axis <= gy_shape.len(),
                        "Bad gradient. You may passed non-scalar value to ag::grad?"
                    );
                    gy_shape.insert(axis, 1);
                }
                gy_view = gy_view.into_shape(gy_shape).unwrap()
            }

            // do broadcast
            if let Some(ret) = gy_view.broadcast(target_shape) {
                ret.to_owned()
            } else {
                panic!("Bad gradient. You may passed non-scalar value to ag::grad?")
            }
        };

        vec![Ok(ret)]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        let sum = ops::reduction_ops::ReduceSum {
            keep_dims: self.should_make_broadcast_dims,
            sparse_axes: self.sparse_axes,
        };
        let axes = inputs[2];
        let gx = Tensor::builder().set_inputs(vec![gy, axes]).build(sum);
        vec![Some(gx), None, None]
    }
}
