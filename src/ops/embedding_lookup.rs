extern crate ndarray;

use std::mem;
use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct EmbeddingLookup {
    pub vec_dim: usize, // dim of embedding vector
}

pub struct EmbeddingLookupGrad {
    pub vec_dim: usize, // dim of embedding vector
}

impl ::Op for EmbeddingLookup {
    fn name(&self) -> &str {
        "EmbeddingLookup"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        // extract inputs
        let table = &xs[1];
        let indices = xs[0].map(|a| *a as usize);

        // make output shape
        let mut x_shape = indices.shape().to_vec();
        x_shape.push(self.vec_dim);
        let output_shape = x_shape.as_slice();

        // make flattened indices
        let flat_vec_indices = indices.into_raw_vec();

        let ret = {
            let flat_indices = flat_vec_indices.as_slice();
            // TODO: consider using parallel iterator of rayon.
            // select is too slow for higher order looking up
            let selected = table.select(ndarray::Axis(0), flat_indices);
            // unwrap is safe
            selected.into_shape(output_shape).unwrap()
        };

        ret
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_op = EmbeddingLookupGrad {
            vec_dim: self.vec_dim,
        };

        vec![None, Some(ops::apply_op(grad_op, &[inputs[0], inputs[1], gy]))]
    }
}

impl ::Op for EmbeddingLookupGrad {
    fn name(&self) -> &str {
        "EmbeddingLookupGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let flat_indices = xs[0].map(|a| *a as usize).into_raw_vec();
        let grad_shape = xs[1].shape();

        let gy: &NdArray = xs[2];
        // reshape gy
        let gy = {
            let shape = (gy.len() / self.vec_dim, self.vec_dim);
            if let Ok(a) = gy.view().into_shape(shape) {
                a
            } else {
                panic!("Incoming gradient for EmbeddingLookup is wrong")
            }
        };

        // init with zeros
        let mut gx = NdArray::zeros(grad_shape);

        // TODO: use parallel iterator
        for (gy_vec, i) in gy.axis_iter(ndarray::Axis(0)).zip(flat_indices) {
            // sliced view
            let mut grad = gx.slice_mut(
                &[
                    ndarray::Si(i as isize, Some(i as isize + 1), 1),
                    ndarray::Si(0, None, 1),
                ],
            );

            // accumulate gradient
            grad.zip_mut_with(&gy_vec, |g, &gy| { *g += gy; });
        }

        gx
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![None, None]
    }
}
