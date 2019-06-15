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

#[test]
fn slice() {
    let ref a: ag::Tensor<f32> = ag::zeros(&[4, 4]);
    let ref b = ag::slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]

    assert_eq!(b.eval(&[]).unwrap().shape(), &[4, 2]);
}
