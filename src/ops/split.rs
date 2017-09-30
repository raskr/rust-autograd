extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Split {
    pub axis: isize,
    pub sizes: Vec<usize>,
    pub index: usize,
}

pub struct SplitGrad {
    pub axis: isize,
    pub sizes: Vec<usize>,
    pub index: usize,
}


impl ops::Op for Split {
    fn name(&self) -> &str
    {
        "Split"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];

        let axis = if self.axis < 0 {
            (x.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let start_index = self.sizes[..self.index].iter().cloned().sum::<usize>() as isize;
        let end_index = start_index + self.sizes[self.index] as isize;
        let indices = make_indices(x, start_index, end_index, axis);
        x.slice(indices.as_slice()).to_owned()
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = SplitGrad {
            axis: self.axis,
            sizes: self.sizes.clone(),
            index: self.index,
        };
        vec![Some(ops::apply_op(op, &[inputs[0], gy]))]
    }
}

impl ops::Op for SplitGrad {
    fn name(&self) -> &str
    {
        "SplitGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let gy = xs[1];
        let mut gx = NdArray::zeros(x.shape());

        let axis = if self.axis < 0 {
            (x.ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        let start_index = self.sizes[..self.index].iter().cloned().sum::<usize>() as isize;
        let end_index = start_index + self.sizes[self.index] as isize;
        let indices = make_indices(x, start_index, end_index, axis);

        gx.slice_mut(indices.as_slice()).zip_mut_with(
            gy,
            |a, &g| *a = g,
        );
        gx
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

#[inline]
fn make_indices(x: &NdArray, start_index: isize, end_index: isize, axis: usize)
    -> Vec<ndarray::Si>
{
    let ndim = x.ndim();
    assert!(ndim > axis, "Wrong split axis");
    (0..ndim)
        .map(|i| {
            if i == axis {
                ndarray::Si(start_index, Some(end_index), 1)
            } else {
                // full slice
                ndarray::Si(0, None, 1)
            }
        })
        .collect::<Vec<ndarray::Si>>()
}
