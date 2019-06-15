extern crate autograd as ag;
extern crate ndarray;

struct MultiOutputOp;

impl ag::op::Op<f32> for MultiOutputOp {
    fn name(&self) -> &str {
        "MultiOutputOp"
    }

    fn compute<'v>(
        &self,
        _: ag::runtime::OpComputeContext<'v, f32>,
    ) -> ag::op::ComputeResults<'v, f32> {
        let a = ag::ndarray_ext::zeros(&[2, 3]);
        let b = ag::ndarray_ext::zeros(&[1, 3]);
        vec![Ok(ag::ArrRepr::Owned(a)), Ok(ag::ArrRepr::Owned(b))]
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

#[test]
fn test_hook() {
    let a: ag::Tensor<f32> = ag::ones(&[4, 2]).p();
    let b: ag::Tensor<f32> = ag::zeros(&[2, 3]).ps();
    let c = ag::matmul(a, b).with_fn(Box::new(|arr| println!("My shape: {:?}", arr.shape())));
    ag::eval(&[c], &[]);
}
