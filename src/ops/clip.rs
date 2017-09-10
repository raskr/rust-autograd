use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct Clip {
    pub min: f32,
    pub max: f32,
}

pub struct ClipGrad {
    pub min: f32,
    pub max: f32,
}

impl ops::Op for Clip {
    fn name(&self) -> &str {
        "Clip"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        xs[0].mapv(move |a| a.min(self.max).max(self.min))
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let op = ops::apply_op(ClipGrad{min: self.min, max: self.max}, &[inputs[0], gy]);
        vec![Some(op)]
    }
}

impl ops::Op for ClipGrad {
    fn name(&self) -> &str {
        "ClipGrad"
    }
    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        // cf. https://github.com/chainer/chainer/blob/master/chainer/functions/math/clip.py
        let mut ret = xs[0].mapv(move |x|
            (((x > self.min) as i32) as f32) * (((x < self.max) as i32) as f32)
        );
        ret *= xs[1];
        ret
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![None, None]
    }
}
