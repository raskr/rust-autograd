extern crate autograd as ag;
extern crate ndarray;


#[test]
fn scalar_add() {
    // graph def
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = 3. + ones + 2;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 6.)
    );
}

#[test]
fn scalar_sub() {
    // graph def
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = 3. - ones - 2;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]))
    );
}

#[test]
fn scalar_mul() {
    // graph def
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = 3. * ones * 2;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 6.)
    );
}

#[test]
fn scalar_div() {
    // graph def
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = 3. / &ones / 2;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.5)
    );
}

#[test]
fn add() {
    // graph def
    let zeros = ag::constant(ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3])));
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = zeros + &ones;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.)
    );
}

#[test]
fn sub() {
    // graph def
    let zeros = ag::constant(ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3])));
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = ones - &zeros;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.)
    );
}

#[test]
fn mul() {
    // graph def
    let zeros = ag::constant(ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3])));
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = zeros * ones;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]))
    );
}

#[test]
fn div() {
    // graph def
    let zeros = ag::constant(ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3])));
    let ones = ag::constant(ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.));
    let z: ag::Tensor = zeros / ones;
    assert_eq!(
        z.eval(),
        ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]))
    );
}
