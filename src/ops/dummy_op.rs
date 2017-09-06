use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;

pub struct DummyOp { pub name: String }
impl ops::Op for DummyOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        panic!("must not be called (DummyOp::lop)")
    }

    fn compute(&mut self, mut xs: &[&::NdArray], _: bool) -> ::NdArray {
        panic!("There exists placeholder(s) couldn't get initial value ({})", self.name)
    }
}

