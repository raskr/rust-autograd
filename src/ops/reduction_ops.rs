extern crate ndarray;

use self::ndarray::Zip;
use ndarray_ext;
use ndarray_ext::NdArray;
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

pub struct ArgMax {
    pub axis: isize,
    pub keep_dim: bool,
}


macro_rules! impl_reduce_forward {
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:expr) => {
        fn $forward_name(
            xs: &[&NdArray],
            keep_dims: bool,
            sparse_axes: bool,
        ) -> Result<NdArray, ::OpComputeErrorStatus>
        {
            let x = xs[0];
            let axes = xs[1];
            if x.shape() == &[] {
                // case of 0 rank
                Ok((*x).clone())
            } else {
                let axes: Vec<usize> = ndarray_ext::axes_as_vec(axes, x.ndim(), sparse_axes);
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

impl ops::Op for ReduceSum {
    fn name(&self) -> &str
    {
        "ReduceSum"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        reduce_sum_forward(xs, self.keep_dims, self.sparse_axes)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ops::array_ops::Broadcast {
            keep_dims: self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        vec![
            Some(ops::apply_op(
                grad_op,
                &[gy, &inputs[0].shape(), inputs[1]],
                None,
            )),
            None,
        ]
    }
}

impl ops::Op for ReduceProd {
    fn name(&self) -> &str
    {
        "ReduceProd"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        reduce_prod_forward(xs, self.keep_dims, self.sparse_axes)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ops::array_ops::Broadcast {
            keep_dims: self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = ops::apply_op(
            grad_op,
            &[&(gy * output), &inputs[0].shape(), inputs[1]],
            None,
        ) / inputs[0];
        vec![Some(gx), None]
    }
}


impl ops::Op for ReduceMin {
    fn name(&self) -> &str
    {
        "ReduceMin"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        reduce_min_forward(xs, self.keep_dims, self.sparse_axes)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        min_max_grad(gy, inputs, output, self.keep_dims, self.sparse_axes)
    }
}

impl ops::Op for ReduceMax {
    fn name(&self) -> &str
    {
        "ReduceMax"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        reduce_max_forward(xs, self.keep_dims, self.sparse_axes)
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
    let grad_op1 = ops::array_ops::Broadcast { keep_dims, sparse_axes };
    let grad_op2 = ops::array_ops::Broadcast { keep_dims, sparse_axes };
    let x = inputs[0];
    let x_shape = inputs[0].shape();
    let y = ops::apply_op(grad_op1, &[output, &x_shape, inputs[1]], None);
    let gy = ops::apply_op(grad_op2, &[gy, &x_shape, inputs[1]], None);
    vec![Some(gy * ops::equal(&x, &y)), None]
}


impl ops::Op for ArgMax {
    fn name(&self) -> &str
    {
        "ArgMax"
    }

    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
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
            Zip::from(&mut mask).and(x).and_broadcast(&maxed).apply(|r,
             a,
             b| {
                *r = ((a == b) as i32) as f32
            });
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

        Ok(result)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl_reduce_forward!(reduce_sum_forward, add, 0.);
impl_reduce_forward!(reduce_min_forward, min, f32::MAX);
impl_reduce_forward!(reduce_max_forward, max, f32::MIN);
impl_reduce_forward!(reduce_prod_forward, mul, 1.);
