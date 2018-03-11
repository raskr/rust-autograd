extern crate autograd as ag;
extern crate ndarray;

#[test]
fn argmax() {
    let ref x = ag::constant(ndarray::arr2(&[[3., 4.], [6., 5.]]));
    let ref y = ag::argmax(x, 1, false);
    assert_eq!(y.eval(&[]), ndarray::arr1(&[1., 0.]).into_dyn());
}

#[test]
fn argmax_with_multi_max_args() {
    let ref x = ag::constant(ndarray::arr1(&[1., 2., 3., 3.]));
    let ref y = ag::argmax(x, 0, false);
    assert_eq!(y.eval(&[]), ndarray::arr0(2.).into_dyn());
}
