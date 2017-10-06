extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct IndexOp {
    pub index: isize,
}

pub struct IndexOpGrad {
    pub index: isize,
}

impl ops::Op for IndexOp {
    fn name(&self) -> &str
    {
        "IndexOp"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x: &NdArray = xs[0];
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let flat_x = x.view().into_shape((x.len())).unwrap();
        if let Some(ret) = flat_x.get(i) {
            NdArray::from_elem(ndarray::IxDyn(&[1]), *ret)
        } else {
            panic!("Index out of bounds");
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let op = IndexOpGrad { index: self.index };
        vec![Some(ops::apply_op(op, &[inputs[0], gy]))]
    }
}

impl ops::Op for IndexOpGrad {
    fn name(&self) -> &str
    {
        "IndexOpGrad"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let gy = xs[1];
        let mut result = NdArray::zeros(x.shape());
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let len = result.len();
        if let Some(a) = result
            .view_mut()
            .into_shape(len)
            .unwrap()  // safe unwrap
            .get_mut(i)
        {
            *a = gy[0];
        } else {
            panic!("Index out of bounds");
        }
        println!("{}", result);
        result
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}
