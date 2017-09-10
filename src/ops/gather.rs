extern crate ndarray;

use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct Gather {
    pub axis: isize,
}

pub struct GatherGrad {
    pub axis: isize,
}

impl ops::Op for Gather {
    fn name(&self) -> &str {
        "Gather"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let indices = xs[0].map(|a| *a as usize);
        let param = &xs[1];
        let param_shape = param.shape();
        let axis = if self.axis == -1 { param.ndim() } else { self.axis as usize };

        let output_shape: Vec<usize> = {
            let former: &[usize] = &param_shape[..axis];
            let latter: &[usize] = &param_shape[axis+1..];
            // doing former + indices.shape() + latter
            former.into_iter().chain(indices.shape()).chain(latter).cloned().collect()
        };

        let flat_indices = indices.into_raw_vec();
        let selected = param.select(ndarray::Axis(axis), flat_indices.as_slice());
        selected.into_shape(output_shape.as_slice()).unwrap()
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_op = GatherGrad {
            axis: self.axis,
        };

        vec![None, Some(ops::apply_op(grad_op, &[inputs[0], inputs[1], gy]))]
    }
}

impl ::Op for GatherGrad {
    fn name(&self) -> &str {
        "GatherGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let indices: &NdArray = xs[0];
        let param: &NdArray = xs[1];
        let param_shape = param.shape();
        let gy: &NdArray = xs[2];
        let axis = if self.axis == -1 { param.ndim() } else { self.axis as usize };

        // get read-only view of gy and reshape it
        let gy = {
            let former = &param_shape[..axis];
            let latter = &param_shape[axis+1..];
            let shape: Vec<usize> =
                former.into_iter().chain(&[indices.len()]).chain(latter).cloned().collect();
            gy.view().into_shape(shape).unwrap()
        };

        let mut gx = NdArray::zeros(param.shape());

        for (gy_sub, &i) in gy.axis_iter(ndarray::Axis(axis)).zip(indices) {
            let i = i as isize;
            // get gx's sub view
            let mut gx_sliced = gx.slice_mut(
                (0..param.ndim()).map(|dim| {
                    if dim == axis {
                        ndarray::Si(i, Some(i+1), 1) // squeezed later
                    } else {
                        ndarray::Si(0, None, 1)
                    }
                }).collect::<Vec<_>>().as_slice()
            );

            // squeeze
            let mut gx_sliced = gx_sliced.remove_axis(ndarray::Axis(axis));
            // assign gy to sliced view
            gx_sliced.zip_mut_with(&gy_sub, |gx, &gy| { *gx += gy; });
        }

        gx
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![None, None, None]
    }
}
