extern crate ndarray;

use tensor::Tensor;
use ops;


pub struct Concat {
    pub axis: usize,
}

pub struct ConcatGrad {
    pub axis: usize,
    pub index: usize,
}

impl ops::Op for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    fn compute(&mut self, xs: &[&::NdArray], _: bool) -> ::NdArray {
        let mut views = vec![];
        for x in xs.iter() {
            views.push(x.view());
        }
        if let Ok(y) = ndarray::stack(ndarray::Axis(self.axis), views.as_slice()) {
            y
        } else {
            panic!("Can't concat arrays whose shapes are incompatible.");
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        // [x1, x2, x3, ..., gy]
        let mut merged_inputs: Vec<&Tensor> = inputs.to_vec();
        merged_inputs.insert(0, gy);
        let merged_inputs: &[&Tensor] = merged_inputs.as_slice();

        let gxs = (0..inputs.len())
            .map(move |i| {
                let grad_op = ConcatGrad {
                    index: i,
                    axis: self.axis as usize,
                };
                Some(ops::apply_op(grad_op, merged_inputs))
            })
            .collect::<Vec<Option<Tensor>>>();
        gxs
    }
}

impl ops::Op for ConcatGrad {
    fn name(&self) -> &str {
        "ConcatGrad"
    }

    fn compute(&mut self, xs: &[&::NdArray], _: bool) -> ::NdArray {
        let gy = xs[0];
        let xs = xs[1..].to_vec();

        // make slice indices
        let mut start_idx = 0;
        for x in xs[..self.index].iter() {
            start_idx += x.shape()[self.axis];
        }
        let region_len = xs[self.index].shape()[self.axis] as isize;
        let indices = (0..gy.ndim())
            .map(move |axis| {
                if axis == self.axis {
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

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}
