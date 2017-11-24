extern crate autograd as ag;
extern crate ndarray;


#[test]
fn scalar_add()
{
    let mut ctx = ag::Context::new();
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]), &mut ctx);
    let ref z: ag::Tensor = 3. + ones + 2;
    assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[6., 6., 6.]).into_dyn());
}

#[test]
fn scalar_sub()
{
    let mut ctx = ag::Context::new();
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]), &mut ctx);
    let ref z: ag::Tensor = 3. - ones - 2;
    assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[0., 0., 0.]).into_dyn());
}

#[test]
fn scalar_mul()
{
    let mut ctx = ag::Context::new();
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]), &mut ctx);
    let ref z: ag::Tensor = 3. * ones * 2;
    assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[6., 6., 6.]).into_dyn());
}

#[test]
fn scalar_div()
{
    let mut ctx = ag::Context::new();
    let ref ones = ag::constant(ndarray::arr1(&[1., 1., 1.]), &mut ctx);
    let ref z: ag::Tensor = 3. / ones / 2;
    assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[1.5, 1.5, 1.5]).into_dyn());
}
