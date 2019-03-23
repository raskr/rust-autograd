extern crate autograd as ag;

struct MultiOutputOp;

impl ag::op::Op<f32> for MultiOutputOp {
    fn name(&self) -> &str {
        "MultiOutputOp"
    }

    fn compute(&self, _: ag::runtime::OpComputeContext<f32>) -> ag::op::ComputeResult<f32> {
        let a = ag::ndarray_ext::zeros(&[2, 3]);
        let b = ag::ndarray_ext::zeros(&[1, 3]);
        vec![Ok(a), Ok(b)]
    }

    fn grad(
        &self,
        _: &ag::Tensor<f32>,
        _: &[&ag::Tensor<f32>],
        _: &ag::Tensor<f32>,
    ) -> Vec<Option<ag::Tensor<f32>>> {
        vec![None; 2]
    }
}

#[test]
fn test_nth_tensor() {
    let ref a = ag::Tensor::builder().build(MultiOutputOp);
    let ref b = ag::nth_tensor(a, 1);
    let ref c = ag::exp(b);
    ag::eval(&[c], &[]);
}
