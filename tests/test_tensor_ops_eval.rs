extern crate autograd as ag;
extern crate ndarray;


#[test]
fn reshape()
{
    let x = ag::zeros(&[3, 2, 2]);
    let y = ag::reshape(&x, &[3, 4]);
    assert_eq!(
        y.eval(),
        x.eval().into_shape(ndarray::IxDyn(&[3, 4])).unwrap()
    );
}
