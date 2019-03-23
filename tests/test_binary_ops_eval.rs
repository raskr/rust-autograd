extern crate autograd as ag;
extern crate ndarray;

#[test]
fn scalar_add() {
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor<f64> = 3. + ones + 2.;
    assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[6., 6., 6.]).into_dyn()));
}

#[test]
fn scalar_sub() {
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor<f64> = 3. - ones - 2.;
    assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[0., 0., 0.]).into_dyn()));
}

#[test]
fn scalar_mul() {
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor<f64> = 3. * ones * 2.;
    assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[6., 6., 6.]).into_dyn()));
}

#[test]
fn scalar_div() {
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor<f64> = 3. / ones / 2.;
    assert_eq!(
        z.eval(&[]),
        Some(ndarray::arr1(&[1.5, 1.5, 1.5]).into_dyn())
    );
}
