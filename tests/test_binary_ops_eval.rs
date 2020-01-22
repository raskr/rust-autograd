extern crate autograd as ag;
extern crate ndarray;

#[test]
fn scalar_add() {
    ag::with(|g| {
        let z: ag::Tensor<f64> = 3. + g.ones(&[3]) + 2.;
        assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[6., 6., 6.]).into_dyn()));
    });
}

#[test]
fn scalar_sub() {
    ag::with(|g| {
        let ref z: ag::Tensor<f64> = 3. - g.ones(&[3]) - 2.;
        assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[0., 0., 0.]).into_dyn()));
    });
}

#[test]
fn scalar_mul() {
    ag::with(|g| {
        let ref z: ag::Tensor<f64> = 3. * g.ones(&[3]) * 2.;
        assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[6., 6., 6.]).into_dyn()));
    });
}

#[test]
fn scalar_div() {
    ag::with(|g| {
        let z: ag::Tensor<f64> = 3. / g.ones(&[3]) / 2.;
        assert_eq!(
            z.eval(&[]),
            Some(ndarray::arr1(&[1.5, 1.5, 1.5]).into_dyn())
        );
    });
}

#[test]
fn slice() {
    ag::with(|g| {
        let ref a: ag::Tensor<f32> = g.zeros(&[4, 4]);
        let ref b = g.slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
        assert_eq!(b.eval(&[]).unwrap().shape(), &[4, 2]);
    });
}
