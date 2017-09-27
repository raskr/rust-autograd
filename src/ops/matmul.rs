extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct MatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl ops::Op for MatMul {
    fn name(&self) -> &str
    {
        "MatMul"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let x0_shape = x0.shape();
        let x1_shape = x1.shape();
        assert_eq!(x0_shape.len(), 2);
        assert_eq!(x1_shape.len(), 2);
        let x0_view = x0.view();
        let x1_view = x1.view();
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
        a.dot(&b).into_dyn()
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let opa = MatMul {
            transpose_a: false,
            transpose_b: true,
        };
        let opb = MatMul {
            transpose_a: true,
            transpose_b: false,
        };
        vec![
            Some(ops::apply_op(opa, &[gy, inputs[1]])),
            Some(ops::apply_op(opb, &[inputs[0], gy])),
        ]
    }
}
