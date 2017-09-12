extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Slice {
    pub indices: Box<[ndarray::Si]>,
}

pub struct SliceGrad {
    pub indices: Box<[ndarray::Si]>,
}

impl ops::Op for Slice {
    fn name(&self) -> &str
    {
        "Slice"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let y: NdArray = xs[0].slice(&*self.indices).to_owned();
        // TODO: for now, if the size of last axis is 1, removing it.
        let last_axis = y.ndim() - 1;
        if y.shape()[last_axis] == 1 {
            y.remove_axis(ndarray::Axis(last_axis))
        } else {
            y
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = SliceGrad { indices: self.indices.clone() };
        vec![Some(ops::apply_op(op, &[inputs[0], gy]))]
    }
}

impl ops::Op for SliceGrad {
    fn name(&self) -> &str
    {
        "SliceGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let gy = xs[1];
        let mut gx = NdArray::zeros(x.shape());
        // sliced view
        gx.slice_mut(&*self.indices).zip_mut_with(
            &gy,
            |a, &g| *a = g,
        );
        gx
    }

    fn lop(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // is this ok?
        vec![None, None]
    }
}
