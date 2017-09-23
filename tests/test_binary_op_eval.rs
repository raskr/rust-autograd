extern crate autograd as ag;
extern crate ndarray;


#[test]
fn scalar_add()
{
    // graph def
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = 3. + ones + 2;
    assert_eq!(z.eval(), ndarray::arr1(&[6., 6., 6.]).into_dyn());
}

#[test]
fn scalar_sub()
{
    // graph def
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = 3. - ones - 2;
    assert_eq!(z.eval(), ndarray::arr1(&[0., 0., 0.]).into_dyn());
}

#[test]
fn scalar_mul()
{
    // graph def
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = 3. * ones * 2;
    assert_eq!(z.eval(), ndarray::arr1(&[6., 6., 6.]).into_dyn());
}

#[test]
fn scalar_div()
{
    // graph def
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = 3. / &ones / 2;
    assert_eq!(z.eval(), ndarray::arr1(&[1.5, 1.5, 1.5]).into_dyn());
}

#[test]
fn add()
{
    // graph def
    let zeros = ag::constant(ndarray::arr1(&[0., 0., 0.]));
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = zeros + &ones;
    assert_eq!(z.eval(), ndarray::arr1(&[1., 1., 1.]).into_dyn());
}

#[test]
fn sub()
{
    // graph def
    let zeros = ag::constant(ndarray::arr1(&[0., 0., 0.]));
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = ones - &zeros;
    assert_eq!(z.eval(), ndarray::arr1(&[1., 1., 1.]).into_dyn());
}

#[test]
fn mul()
{
    // graph def
    let zeros = ag::constant(ndarray::arr1(&[0., 0., 0.]));
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = zeros * ones;
    assert_eq!(z.eval(), ndarray::arr1(&[0., 0., 0.]).into_dyn());
}

#[test]
fn div()
{
    // graph def
    let zeros = ag::constant(ndarray::arr1(&[0., 0., 0.]));
    let ones = ag::constant(ndarray::arr1(&[1., 1., 1.]));
    let z: ag::Tensor = zeros / ones;
    assert_eq!(z.eval(), ndarray::arr1(&[0., 0., 0.]).into_dyn());
}
