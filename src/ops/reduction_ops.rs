extern crate ndarray;

use self::ndarray::Zip;
use ndarray_ext;
use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct ArgMax {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceMin {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceMax {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceSum {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceMean {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceMinGrad {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceMaxGrad {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceSumGrad {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceMeanGrad {
    pub axis: isize,
    pub keep_dim: bool,
}



impl ops::Op for ArgMax {
    fn name(&self) -> &str
    {
        "ArgMax"
    }

    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let axis = if -1 == self.axis {
            x.ndim() - 1
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

        result
    }

    fn lop(&self, _:&Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}


impl ops::Op for ReduceMin {
    fn name(&self) -> &str
    {
        "ReduceMin"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];

        let axis = if -1 == self.axis {
            x.ndim() - 1
        } else {
            self.axis as usize
        };

        let min_fn = f32::min;
        let min = x.fold_axis(ndarray::Axis(axis), f32::MAX, move |&a, &b| min_fn(a, b));

        if self.keep_dim {
            ndarray_ext::expand_dims(min, axis)
        } else {
            min
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ReduceMinGrad {
            axis: self.axis,
            keep_dim: self.keep_dim,
        };
        vec![Some(ops::apply_op(grad_op, &[inputs[0], output, gy]))]
    }
}

impl ops::Op for ReduceMinGrad {
    fn name(&self) -> &str
    {
        "ReduceMinGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0].view();
        let y = xs[1].view();
        let gy = xs[2].view();
        let axis = if -1 == self.axis {
            x.ndim() - 1
        } else {
            self.axis as usize
        };

        let (y, gy) = if self.keep_dim {
            (y, gy)
        } else {
            (
                ndarray_ext::expand_dims_view(y, axis),
                ndarray_ext::expand_dims_view(gy, axis),
            )
        };

        // compare x and y
        let mut mask = NdArray::zeros(x.shape());
        Zip::from(&mut mask).and(x).and_broadcast(&y).apply(|r, a, b|
            *r = ((a == b) as i32) as f32
        );

        mask *= &gy;
        mask
    }


    fn lop(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None, None]
    }
}

impl ops::Op for ReduceMax {
    fn name(&self) -> &str
    {
        "ReduceMax"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];

        let axis = if -1 == self.axis {
            x.ndim() - 1
        } else {
            self.axis as usize
        };

        let max_fn = f32::max;
        let maxed = x.fold_axis(ndarray::Axis(axis), f32::MIN, move |&a, &b| max_fn(a, b));

        if self.keep_dim {
            ndarray_ext::expand_dims(maxed, axis)
        } else {
            maxed
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ReduceMaxGrad {
            axis: self.axis,
            keep_dim: self.keep_dim,
        };
        vec![Some(ops::apply_op(grad_op, &[inputs[0], output, gy]))]
    }
}

impl ops::Op for ReduceMaxGrad {
    fn name(&self) -> &str
    {
        "ReduceMaxGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0].view();
        let y = xs[1].view();
        let gy = xs[2].view();
        let axis = if -1 == self.axis {
            x.ndim() - 1
        } else {
            self.axis as usize
        };

        let (y, gy) = if self.keep_dim {
            (y, gy)
        } else {
            (
                ndarray_ext::expand_dims_view(y, axis),
                ndarray_ext::expand_dims_view(gy, axis),
            )
        };

        // compare x and y
        let mut mask = NdArray::zeros(x.shape());
        Zip::from(&mut mask).and(x).and_broadcast(&y).apply(|r, a, b|
            *r = ((a == b) as i32) as f32
        );

        mask *= &gy;
        mask
    }

    fn lop(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None, None]
    }
}

impl ops::Op for ReduceMean {
    fn name(&self) -> &str
    {
        "ReduceMean"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x: &NdArray = xs[0];

        // TODO
        if 1 == x.ndim() {
            panic!(
                "ReduceMean: input of row vector is not supported.\
                Consider converting it to vertical vector."
            )
        }

        let axis = if -1 == self.axis {
            x.ndim()
        } else {
            self.axis as usize
        };

        if self.keep_dim {
            let ret = x.mean(ndarray::Axis(axis));
            ndarray_ext::expand_dims(ret, axis)
        } else {
            x.mean(ndarray::Axis(axis))
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ReduceMeanGrad {
            axis: self.axis,
            keep_dim: self.keep_dim,
        };
        vec![Some(ops::apply_op(grad_op, &[inputs[0], gy]))]
    }
}


impl ops::Op for ReduceMeanGrad {
    fn name(&self) -> &str
    {
        "ReduceMeanGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let axis = if -1 == self.axis {
            x.ndim() - 1
        } else {
            self.axis as usize
        };

        // add dim for broadcast (the result is "view")
        let gy = if self.keep_dim {
            xs[1].view()
        } else {
            ndarray_ext::expand_dims_view(xs[1].view(), axis)
        };

        // do broadcast and division
        if let Some(gx) = gy.broadcast(x.shape()) {
            let mut gx = gx.to_owned();
            let reduction_len = x.shape()[axis] as f32;
            gx /= reduction_len;
            gx
        } else {
            panic!("Lop implementation of immediate successor of ReduceMean is wrong")
        }
    }

    fn lop(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}

impl ops::Op for ReduceSum {
    fn name(&self) -> &str
    {
        "ReduceSum"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = &xs[0];

        let axis = if -1 == self.axis {
            x.ndim()
        } else {
            self.axis as usize
        };

        if self.keep_dim {
            let ret = x.sum(ndarray::Axis(axis));
            ndarray_ext::expand_dims(ret, axis)
        } else {
            x.sum(ndarray::Axis(axis))
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let grad_op = ReduceSumGrad {
            axis: self.axis,
            keep_dim: self.keep_dim,
        };
        vec![Some(ops::apply_op(grad_op, &[inputs[0], gy]))]
    }
}

impl ops::Op for ReduceSumGrad {
    fn name(&self) -> &str
    {
        "ReduceSumGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let axis = if -1 == self.axis {
            x.ndim() - 1
        } else {
            self.axis as usize
        };

        // add dim for broadcast (the result is "view")
        let gy = if self.keep_dim {
            xs[1].view()
        } else {
            ndarray_ext::expand_dims_view(xs[1].view(), axis)
        };

        // do broadcast
        if let Some(gx) = gy.broadcast(x.shape()) {
            gx.to_owned()
        } else {
            panic!("Lop implementation of immediate successor of ReduceSum is wrong")
        }
    }

    fn lop(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
