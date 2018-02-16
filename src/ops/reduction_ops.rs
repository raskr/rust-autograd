extern crate ndarray;

use self::ndarray::Zip;
use ndarray_ext;
use ndarray_ext::NdArray;
use op;
use ops;
use std::f32;
use std::mem;
use std::ops::Add;
use std::ops::Mul;
use tensor::Tensor;

pub struct ReduceMin
{
    pub keep_dims:   bool,
    pub sparse_axes: bool,
}

pub struct ReduceMax
{
    pub keep_dims:   bool,
    pub sparse_axes: bool,
}

pub struct ReduceProd
{
    pub keep_dims:   bool,
    pub sparse_axes: bool,
}

pub struct ReduceSum
{
    pub keep_dims:   bool,
    pub sparse_axes: bool,
}

pub struct ReduceMean
{
    pub keep_dims:   bool,
    pub sparse_axes: bool,
}

pub struct ArgMax
{
    pub axis:     isize,
    pub keep_dim: bool,
}

macro_rules! impl_reduce_forward {
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:expr) => {
        fn $forward_name(
            x: &NdArray,
            axes: &NdArray,
            should_preprocess_axes: bool,
            keep_dims: bool,
            sparse_axes: bool,
        ) -> Result<NdArray, ::errors::OpComputeErrorStatus>
        {
            let x_shape = x.shape();
            if x_shape == &[] {
                // case of 0 rank
                Ok((*x).clone())
            } else {
                let axes: Vec<usize> = if should_preprocess_axes {
                    ndarray_ext::axes_as_vec(axes, x_shape.len(), sparse_axes)
                } else {
                    axes.iter().map(|&a| a as usize).collect()
                };
                if axes.is_empty() {
                    return Err(::OpComputeErrorStatus::Delegate { to: 0 });
                }

                let mut folded: Option<NdArray> = None;

                for axis in axes.into_iter() {
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

impl_reduce_forward!(comopute_reduce_sum, add, 0.);
impl_reduce_forward!(comopute_reduce_min, min, f32::MAX);
impl_reduce_forward!(comopute_reduce_max, max, f32::MIN);
impl_reduce_forward!(comopute_reduce_prod, mul, 1.);

impl op::Op for ReduceSum
{
    fn name(&self) -> &str
    {
        "ReduceSum"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        vec![
            comopute_reduce_sum(xs[0], xs[1], true, self.keep_dims, self.sparse_axes),
        ]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ops::array_ops::Broadcast {
            keep_dims:   self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder()
            .set_inputs(vec![gy, &inputs[0].shape(), inputs[1]])
            .build(grad_op);
        vec![Some(gx), None]
    }
}

impl op::Op for ReduceMean
{
    fn name(&self) -> &str
    {
        "ReduceMean"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let sum = comopute_reduce_sum(xs[0], xs[1], false, self.keep_dims, self.sparse_axes);
        vec![
            sum.map(|mut ok| {
                let x_shape = xs[0].shape();
                let reduction_axes: Vec<usize> = xs[1].iter().map(|&a| a as usize).collect();
                let reduction_len: f32 = reduction_axes
                    .into_iter()
                    .map(|i| x_shape[i] as f32)
                    .product();
                ok *= 1. / reduction_len;
                ok
            }),
        ]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ops::array_ops::Broadcast {
            keep_dims:   self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let x = inputs[0];
        let axes = inputs[1]; // this is preprocessed
        let ref reduction_len = ops::reduce_prod(&ops::gather(&x.shape(), axes, 0), &[0], false);
        let tmp = Tensor::builder()
            .set_inputs(vec![gy, &inputs[0].shape(), inputs[1]])
            .build(grad_op);
        let gx = tmp * ops::reciprocal(reduction_len);
        vec![Some(gx), None]
    }
}

impl op::Op for ReduceProd
{
    fn name(&self) -> &str
    {
        "ReduceProd"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        vec![
            comopute_reduce_prod(xs[0], xs[1], true, self.keep_dims, self.sparse_axes),
        ]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ops::array_ops::Broadcast {
            keep_dims:   self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let tmp = Tensor::builder()
            .set_inputs(vec![&(gy * output), &inputs[0].shape(), inputs[1]])
            .build(grad_op);
        let gx = tmp / inputs[0];
        vec![Some(gx), None]
    }
}

impl op::Op for ReduceMin
{
    fn name(&self) -> &str
    {
        "ReduceMin"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        vec![
            comopute_reduce_min(xs[0], xs[1], true, self.keep_dims, self.sparse_axes),
        ]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

impl op::Op for ReduceMax
{
    fn name(&self) -> &str
    {
        "ReduceMax"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        vec![
            comopute_reduce_max(xs[0], xs[1], true, self.keep_dims, self.sparse_axes),
        ]
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

fn min_max_grad(
    gy: &Tensor,
    inputs: &[&Tensor],
    output: &Tensor,
    keep_dims: bool,
    sparse_axes: bool,
) -> Vec<Option<Tensor>>
{
    let grad_op1 = ops::array_ops::Broadcast {
        keep_dims,
        sparse_axes,
    };
    let grad_op2 = ops::array_ops::Broadcast {
        keep_dims,
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

impl op::Op for ArgMax
{
    fn name(&self) -> &str
    {
        "ArgMax"
    }

    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: ::runtime::OpComputeContext) -> op::ComputeResult
    {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let axis = if self.axis < 0 {
            (x.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };
        let x_shape = x.shape();

        // 1. Make binary mask tensor (maximum is 1)
        let mut mask = {
            let max_fn = f32::max;
            let maxed = x.fold_axis(ndarray::Axis(axis), f32::MIN, move |&a, &b| max_fn(a, b));
            let maxed = ndarray_ext::expand_dims(maxed, axis);
            let mut mask = NdArray::zeros(x.shape());
            Zip::from(&mut mask)
                .and(x)
                .and_broadcast(&maxed)
                .apply(|r, a, b| *r = ((a == b) as i32) as f32);
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

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}
