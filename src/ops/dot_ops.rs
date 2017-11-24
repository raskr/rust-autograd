/// `Tensordot` is implemented in `ops/mod.rs`.
extern crate ndarray;
extern crate rayon;

use self::rayon::iter::*;
use ndarray_ext;
use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct MatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct BatchMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}


impl ops::Op for MatMul {
    fn name(&self) -> &str
    {
        "MatMul"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let x0_shape = x0.shape();
        let x1_shape = x1.shape();
        if x0_shape.len() != 2 {
            return Err(::OpComputeErrorStatus::BadInput(
                "First input to the matmul should be Matrix".to_string(),
            ));
        }
        if x1_shape.len() != 2 {
            return Err(::OpComputeErrorStatus::BadInput(
                "Second input to the matmul should be Matrix".to_string(),
            ));
        }
        let x0_view = x0.view();
        let x1_view = x1.view();
        // unwrap is always safe
        let mut a = x0_view.into_shape((x0_shape[0], x0_shape[1])).unwrap();
        let mut b = x1_view.into_shape((x1_shape[0], x1_shape[1])).unwrap();
        if self.transpose_a {
            // almost zero cost
            a.swap_axes(0, 1);
        }
        if self.transpose_b {
            // almost zero cost
            b.swap_axes(0, 1);
        }
        Ok(a.dot(&b).into_dyn())
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let opa = MatMul { transpose_a: false, transpose_b: true };
        let opb = MatMul { transpose_a: true, transpose_b: false };
        vec![
            Some(ops::apply_op(opa, &[gy, inputs[1]], None)),
            Some(ops::apply_op(opb, &[inputs[0], gy], None)),
        ]
    }
}


impl ops::Op for BatchMatMul {
    fn name(&self) -> &str
    {
        "BatchMatMul"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0: &NdArray = xs[0];
        let x1: &NdArray = xs[1];
        let shape0 = x0.shape();
        let shape1 = x1.shape();
        let rank0 = x0.ndim();
        let rank1 = x1.ndim();

        if rank0 != rank1 || shape0[..rank0 - 2] != shape1[..rank0 - 2] {
            return Err(::OpComputeErrorStatus::BadInput(format!(
                "Input shape mismatch: {:?} vs {:?}",
                shape0,
                shape1
            )));
        }

        let row0 = shape0[rank0 - 2];
        let row1 = shape1[rank0 - 2];

        let col0 = shape0[rank0 - 1];
        let col1 = shape1[rank0 - 1];

        // squashes dims (remains last two dims)
        // unwrap is always safe
        let x0_flattened = {
            let mut a = x0.view()
                .into_shape((x0.len() / row0 / col0, row0, col0))
                .unwrap();
            if self.transpose_a {
                a.swap_axes(1, 2);
            }
            a
        };

        let x1_flattened = {
            let mut b = x1.view()
                .into_shape((x1.len() / row1 / col1, row1, col1))
                .unwrap();
            if self.transpose_b {
                b.swap_axes(1, 2);
            }
            b
        };

        // parallel dot
        let dot = (0..x0_flattened.shape()[0] as isize)
            .into_par_iter()
            .map(|i| {
                let x0_mat = x0_flattened
                    .slice(s![i..i + 1, .., ..])
                    .remove_axis(ndarray::Axis(0))
                    .to_owned();
                let x1_mat = x1_flattened
                    .slice(s![i..i + 1, .., ..])
                    .remove_axis(ndarray::Axis(0))
                    .to_owned();
                x0_mat.dot(&x1_mat).into_dyn()
            })
            .collect::<Vec<_>>();

        // owned to ref
        let mut dot_view = Vec::with_capacity(dot.len());
        for i in 0..dot.len() {
            dot_view.push(ndarray_ext::expand_dims_view(dot[i].view(), 0));
        }

        // stack dot result
        let stacked = ndarray::stack(ndarray::Axis(0), dot_view.as_slice()).unwrap();

        let dst_shape = {
            let stacked_shape = stacked.shape();
            shape0[..rank0 - 2]
                .into_iter()
                .chain(&[stacked_shape[1], stacked_shape[2]])
                .cloned()
                .collect::<Vec<usize>>()
        };

        // reshape to dst shape
        Ok(
            stacked
                .into_shape(ndarray::IxDyn(dst_shape.as_slice()))
                .unwrap(),
        ) // safe unwrap

    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // Using `ops::swap_axes()` here causes "invalid memory layout error"
        // so transpose_* flags is used in BatchMatMul.
        let opa = BatchMatMul { transpose_a: false, transpose_b: true };
        let opb = BatchMatMul { transpose_a: true, transpose_b: false };
        vec![
            Some(ops::apply_op(opa, &[gy, inputs[1]], None)),
            Some(ops::apply_op(opb, &[inputs[0], gy], None)),
        ]
    }
}
