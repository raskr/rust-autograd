use ops;
use tensor::Tensor;

pub struct DummyOp {
    pub name: String,
}
impl ops::Op for DummyOp {
    fn name(&self) -> &str
    {
        &self.name
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        panic!("must not be called (DummyOp::grad)")
    }

    fn compute(&mut self, _: &[&::NdArray], _: bool) -> ::NdArray
    {
        panic!(
            "There exists placeholder(s) couldn't get initial value",
            self.name
        )
    }
}
