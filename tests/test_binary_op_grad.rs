extern crate autograd as ag;
extern crate ndarray;



#[test]
fn scalar_add()
{
    let ref x = ag::placeholder(&[]);
    let ref y = x + 2;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    assert_eq!(1., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn scalar_sub()
{
    let ref x = ag::placeholder(&[]);
    let ref y = x - 2;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    assert_eq!(1., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn scalar_mul()
{
    let ref x = ag::placeholder(&[]);
    let ref y = 3 * x;
    let ref grad = ag::grad(&[y], &[x])[0];
    assert_eq!(3., grad.eval(&mut ag::Context::new())[ndarray::IxDyn(&[])]);
}

#[test]
fn scalar_div()
{
    let ref x = ag::placeholder(&[]);
    let ref y = x / 3;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    assert_eq!(1. / 3., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn expr1()
{
    let ref x = ag::placeholder(&[]);
    let ref y = 3 * x + 2;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    assert_eq!(3., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn expr2()
{
    let ref x = ag::placeholder(&[]);
    let ref y = 3 * x * x;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    ctx.feed_input(x, ndarray::arr0(3.));
    assert_eq!(18., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn expr3()
{
    let ref x = ag::placeholder(&[]);
    let ref y = 3 * x * x + 2;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    ctx.feed_input(x, ndarray::arr0(3.));
    assert_eq!(18., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn expr4()
{
    let ref x = ag::placeholder(&[]);
    let ref y = 3 * x * x + 2 * x + 1;
    let ref grad = ag::grad(&[y], &[x])[0];
    let mut ctx = ag::Context::new();
    ctx.feed_input(x, ndarray::arr0(3.));
    assert_eq!(20., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn expr5()
{
    let ref x1 = ag::placeholder(&[]);
    let ref x2 = ag::placeholder(&[]);
    let ref y = 3 * x1 * x1 + 2 * x1 + x2 + 1;
    let ref grad = ag::grad(&[y], &[x1])[0];
    let mut ctx = ag::Context::new();
    ctx.feed_input(x1, ndarray::arr0(3.));
    assert_eq!(20., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
// Test with intention that grad of `x2` should be computed
// even if the value of `x1` is not given
fn expr6()
{
    let mut ctx = ag::Context::new();
    let ref x1 = ag::placeholder(&[]);
    let ref x2 = ag::variable(ndarray::arr0(0.), &mut ctx);
    let ref y = 3 * x1 * x1 + 5 * x2;
    let ref grad = ag::grad(&[y], &[x2])[0];
    assert_eq!(5., grad.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}

#[test]
fn differentiate_twice()
{
    let ref x = ag::placeholder(&[]);
    let ref y = x * x;
    let ref g1 = ag::grad(&[y], &[x])[0];
    let ref g2 = ag::grad(&[g1], &[x])[0];
    let mut ctx = ag::Context::new();
    assert_eq!(2., g2.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}


#[test]
fn expr7()
{
    let ref x1 = ag::placeholder(&[]);
    let ref x2 = ag::placeholder(&[]);
    let ref y = 2 * x1 * x1 + 3 * x2;
    let ref g1 = ag::grad(&[y], &[x1])[0];
    let ref g2 = ag::grad(&[y], &[x2])[0];
    let ref gg1 = ag::grad(&[g1], &[x1])[0];

    let mut ctx = ag::Context::new();
    assert_eq!(3., g2.eval(&mut ctx)[ndarray::IxDyn(&[])]);
    assert_eq!(4., gg1.eval(&mut ctx)[ndarray::IxDyn(&[])]);
    ctx.feed_input(x1, ndarray::arr0(2.));
    assert_eq!(8., g1.eval(&mut ctx)[ndarray::IxDyn(&[])]);
}
