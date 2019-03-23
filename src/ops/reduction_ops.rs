use ndarray;
use ndarray_ext;
use ndarray_ext::NdArray;
use op;
use ops;
use std::f32;
use std::mem;
use tensor::Tensor;
use Float;

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
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:ident) => {
        fn $forward_name<T: Float>(
            x: &NdArray<T>,
            mut axes: Vec<usize>,
            keep_dims: bool,
        ) -> Result<NdArray<T>, ::op::ComputeException> {
            let x_shape = x.shape();

            if ndarray_ext::is_scalar_shape(x_shape) {
                // case of 0 rank
                Ok((*x).clone())
            } else {
                // reduction axes are empty => do nothing
                if axes.is_empty() {
                    return Err(::op::ComputeException::Delegate { to: 0 });
                }

                // -- main logic --
                let mut folded: Option<NdArray<T>> = None;
                axes.sort();

                for axis in axes.into_iter().rev() {
                    let func = T::$reduce_fn_name;
                    let ret = folded.as_ref().unwrap_or(x).fold_axis(
                        ndarray::Axis(axis),
                        T::$reduce_default(),
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

impl_reduce_forward!(compute_reduce_sum, add, zero);
impl_reduce_forward!(compute_reduce_min, min, max_value);
impl_reduce_forward!(compute_reduce_max, max, min_value);
impl_reduce_forward!(compute_reduce_prod, mul, one);

#[inline]
fn preprocess_axes<T: Float>(x: &NdArray<T>, axes: &NdArray<T>, sparse_axes: bool) -> Vec<usize> {
    if sparse_axes {
        ndarray_ext::sparse_to_dense(axes)
    } else {
        ndarray_ext::normalize_negative_axes(axes, x.ndim())
    }
}

impl<T: Float> op::Op<T> for ReduceSum {
    fn name(&self) -> &str {
        "ReduceSum"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        vec![compute_reduce_sum(x, axes, self.keep_dims)]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
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

impl<T: Float> op::Op<T> for ReduceMean {
    fn name(&self) -> &str {
        "ReduceMean"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        let x_shape = x.shape();
        if axes.is_empty() {
            return vec![Err(::op::ComputeException::Delegate { to: 0 })];
        }

        // Make reduction_len
        let mut reduction_len = 1.;
        for &axis in axes.iter() {
            reduction_len *= x_shape[axis as usize] as f32;
        }
        let reduction_len_inv = T::one() / T::from(reduction_len).unwrap();

        // Do summation
        let sum: Result<NdArray<T>, _> = compute_reduce_sum(x, axes, self.keep_dims);

        // Do division
        let ret = sum.map(|mut ok| {
            ok.mapv_inplace(move |elem| elem * reduction_len_inv);
            ok
        });

        vec![ret]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
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
        let reduction_len = &ops::reduce_prod(reduction_sizes, &[0], false);
        let gx = broadcast / reduction_len;

        vec![Some(gx), None]
    }
}

impl<T: Float> op::Op<T> for ReduceProd {
    fn name(&self) -> &str {
        "ReduceProd"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        let ret = compute_reduce_prod(x, axes, self.keep_dims);
        vec![ret]
    }

    fn grad(
        &self,
        gy: &Tensor<T>,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<Option<Tensor<T>>> {
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

impl<T: Float> op::Op<T> for ReduceMin {
    fn name(&self) -> &str {
        "ReduceMin"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        vec![compute_reduce_min(x, axes, self.keep_dims)]
    }

    fn grad(
        &self,
        gy: &Tensor<T>,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<Option<Tensor<T>>> {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

impl<T: Float> op::Op<T> for ReduceMax {
    fn name(&self) -> &str {
        "ReduceMax"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axes = preprocess_axes(x, xs[1], self.sparse_axes);
        vec![compute_reduce_max(x, axes, self.keep_dims)]
    }

    fn grad(
        &self,
        gy: &Tensor<T>,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<Option<Tensor<T>>> {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

fn min_max_grad<T: Float>(
    gy: &Tensor<T>,
    inputs: &[&Tensor<T>],
    output: &Tensor<T>,
    keep_dims: bool,
    sparse_axes: bool,
) -> Vec<Option<Tensor<T>>> {
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

impl<T: Float> op::Op<T> for ArgMax {
    fn name(&self) -> &str {
        "ArgMax"
    }

    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let x_shape = x.shape();

        // 1. Make binary mask tensor (maximums are 1s)
        let mut mask = {
            let max_fn = T::max;
            let min_val = T::min_value();
            let maxed = x.fold_axis(ndarray::Axis(axis), min_val, move |&a, &b| max_fn(a, b));
            let mut mask = x.clone();
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

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for ReduceGradCommon {
    fn name(&self) -> &str {
        "ReduceGradCommon"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
        let xs = ctx.grab_inputs();
        //  broadcast `gy` into `target_shape`
        let gy = xs[0];
        let target_shape = ndarray_ext::vec_as_shape(xs[1]); // x's shape

        if gy.shape() == target_shape.as_slice() {
            return vec![Err(::op::ComputeException::Delegate { to: 0 })];
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
                        "Bad gradient. You may passed a non-scalar value to `ag::grad`?"
                    );
                    gy_shape.insert(axis, 1);
                }
                gy_view = gy_view.into_shape(gy_shape).unwrap()
            }

            // do broadcast
            if let Some(ret) = gy_view.broadcast(target_shape) {
                ret.to_owned()
            } else {
                panic!("Bad gradient. You may passed a non-scalar value to `ag::grad`?")
            }
        };

        vec![Ok(ret)]
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let sum = ops::reduction_ops::ReduceSum {
            keep_dims: self.should_make_broadcast_dims,
            sparse_axes: self.sparse_axes,
        };
        let axes = inputs[2];
        let gx = Tensor::builder().set_inputs(vec![gy, axes]).build(sum);
        vec![Some(gx), None, None]
    }
}
