use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;

pub struct ELU {
    pub alpha: f32,
}
pub struct ELUGrad {
    pub alpha: f32,
}

impl ops::Op for ELU {
    fn name(&self) -> &str
    {
        "ELU"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].mapv(move |a| if a > 0. {
            a
        } else {
            self.alpha * (a.exp() - 1.)
        })
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let gx = ops::apply_op(ELUGrad { alpha: self.alpha }, &[inputs[0], gy]);
        vec![Some(gx)]
    }
}

impl ops::Op for ELUGrad {
    fn name(&self) -> &str
    {
        "ELUGrad"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let gy = xs[1];
        let a = x.mapv(move |a| if a > 0. {
            1.
        } else {
            self.alpha * (a.exp() - 1.) + self.alpha
        });
        a * gy
    }

    // TODO: impl
    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
