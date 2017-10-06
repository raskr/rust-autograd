extern crate ndarray;

use ops;
use tensor::Tensor;


pub struct Concat {
    pub axis: isize,
}

pub struct ConcatGrad {
    pub axis: isize,
    pub index: usize,
}

impl ops::Op for Concat {
    fn name(&self) -> &str
    {
        "Concat"
    }

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        let mut views = vec![];
        for x in xs.iter() {
            views.push(x.view());
        }

        let axis = if self.axis < 0 {
            (xs[0].ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        if let Ok(y) = ndarray::stack(ndarray::Axis(axis), views.as_slice()) {
            y
        } else {
            panic!("Can't concat arrays whose shapes are incompatible.");
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // [x1, x2, x3, ..., gy]
        let mut merged_inputs: Vec<&Tensor> = inputs.to_vec();
        merged_inputs.insert(0, gy);
        let merged_inputs: &[&Tensor] = merged_inputs.as_slice();

        let gxs = (0..inputs.len())
            .map(move |i| {
                let grad_op = ConcatGrad {
                    index: i,
                    axis: self.axis,
                };
                Some(ops::apply_op(grad_op, merged_inputs))
            })
            .collect::<Vec<Option<Tensor>>>();
        gxs
    }
}

impl ops::Op for ConcatGrad {
    fn name(&self) -> &str
    {
        "ConcatGrad"
    }

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        let gy = xs[0];
        let xs = xs[1..].to_vec();

        let axis = if self.axis < 0 {
            (xs[0].ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        // make slice indices
        let mut start_idx = 0;
        for x in xs[..self.index].iter() {
            start_idx += x.shape()[axis];
        }
        let region_len = xs[self.index].shape()[axis] as isize;
        let indices = (0..gy.ndim())
            .map(move |_axis| {
                if _axis == axis {
                    // partial region
                    ndarray::Si(start_idx as isize, Some(region_len), 1)
                } else {
                    // full slice
                    ndarray::Si(0, None, 1)
                }
            })
            .collect::<Vec<ndarray::Si>>();

        // do slice
        gy.slice(&*indices).to_owned()
    }

    fn grad(&self, _: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        (0..inputs.len()).map(|_| None).collect::<Vec<_>>()
    }
}
