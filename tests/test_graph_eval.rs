extern crate autograd as ag;
extern crate ndarray;


#[test]
fn const_eval()
{
    let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[2]));
    let a = ag::constant(arr.clone());
    assert_eq!(arr, a.eval());
}

#[test]
fn variable_eval()
{
    let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[2]));
    let a = ag::variable(arr.clone());
    assert_eq!(arr, a.eval());
}
