extern crate autograd as ag;
extern crate ndarray;



#[test]
fn scalar_add()
{
    let ref x = ag::variable(ndarray::arr1(&[]));
    let ref y = x + 2;
    let grads = ag::gradients(y, &[x], None);
    assert_eq!(1., grads[0].eval()[0]);
}

#[test]
fn scalar_sub()
{
    let ref x = ag::variable(ndarray::arr1(&[]));
    let ref y = x - 2;
    let grads = ag::gradients(y, &[x], None);
    assert_eq!(1., grads[0].eval()[0]);
}

#[test]
fn scalar_mul()
{
    let ref x = ag::variable(ndarray::arr1(&[]));
    let ref y = 3 * x;
    let grads = ag::gradients(y, &[x], None);
    assert_eq!(3., grads[0].eval()[0]);
}

#[test]
fn scalar_div()
{
    let ref x = ag::variable(ndarray::arr1(&[]));
    let ref y = x / 3;
    let grads = ag::gradients(y, &[x], None);
    assert_eq!(1. / 3., grads[0].eval()[0]);
}

#[test]
fn expr1()
{
    let ref x = ag::variable(ndarray::arr1(&[]));
    let ref y = 3 * x + 2;
    let grads = ag::gradients(y, &[x], None);
    assert_eq!(3., grads[0].eval()[0]);
}

#[test]
fn expr2()
{
    let ref x = ag::placeholder();
    let ref y = 3 * x * x;
    let grads = ag::gradients(y, &[x], None);
    let fd = ag::Feed::new().add(
        x,
        ndarray::arr1(&[3.]),
    );
    assert_eq!(18., grads[0].eval_with_input(fd)[0]);
}

#[test]
fn expr3()
{
    let ref x = ag::placeholder();
    let ref y = 3 * x * x + 2;
    let grads = ag::gradients(y, &[x], None);
    let fd = ag::Feed::new().add(
        x,
        ndarray::arr1(&[3.]),
    );
    assert_eq!(18., grads[0].eval_with_input(fd)[0]);
}

#[test]
fn expr4()
{
    let ref x = ag::placeholder();
    let ref y = 3 * x * x + 2 * x + 1;
    let grads = ag::gradients(y, &[x], None);
    let fd = ag::Feed::new().add(
        x,
        ndarray::arr1(&[3.]),
    );
    assert_eq!(20., grads[0].eval_with_input(fd)[0]);
}

#[test]
fn expr5()
{
    let ref x1 = ag::placeholder();
    let ref x2 = ag::placeholder();
    let ref y = 3 * x1 * x1 + 2 * x1 + x2 + 1;
    let grads = ag::gradients(y, &[x1], None);
    let fd = ag::Feed::new().add(
        x1,
        ndarray::arr1(&[3.]),
    );
    assert_eq!(20., grads[0].eval_with_input(fd)[0]);
}

#[test]
// Test with intention that grad of `x2` should be computed
// even if the value of `x1` is not given
fn expr6()
{
    let ref x1 = ag::placeholder();
    let ref x2 = ag::variable(ndarray::arr1(&[]));
    let ref y = 3 * x1 * x1 + 5 * x2 + 1;
    let grads = ag::gradients(y, &[x2], None);
    assert_eq!(5., grads[0].eval()[0]);
}

#[test]
fn differentiate_twice()
{
    let ref x = ag::placeholder();
    let ref y = x * x;
    let ref g1 = ag::gradients(y, &[x], None)[0];
    let ref g2 = ag::gradients(g1, &[x], None)[0];

    let fd = ag::Feed::new().add(
        x,
        ndarray::arr1(&[2.]),
    );

    assert_eq!(2., g2.eval_with_input(fd)[0]);
}


#[test]
fn expr7()
{
    let ref x1 = ag::placeholder();
    let ref x2 = ag::variable(ag::ndarray_ext::zeros(&[1]));
    let ref y = 2 * x1 * x1 + 3 * x2 + 1;
    let ref g1 = ag::gradients(y, &[x1], None)[0];
    let ref g2 = ag::gradients(y, &[x2], None)[0];
    let ref gg1 = ag::gradients(g1, &[x1], None)[0];

    assert_eq!(
        8.,
        g1.eval_with_input(ag::Feed::new().add(x1, ndarray::arr1(&[2.])))[0]
    ); // => [8.]
    assert_eq!(3., g2.eval()[0]); // => [3.]
    assert_eq!(4., gg1.eval()[0]); // => [4.]
}

#[test]
fn expr8()
{
    let ref x = ag::placeholder();
    let ref y = ag::variable(ag::ndarray_ext::zeros(&[1]));
    let ref z = 2 * x * x + 3 * y + 1;
    let ref g1 = ag::gradients(z, &[y], None)[0];
    let ref g2 = ag::gradients(z, &[x], None)[0];
    let ref gg = ag::gradients(g2, &[x], None)[0];

    // dz/dy
    assert_eq!(3., g1.eval()[0]);

    // dz/dx (necessary to feed the value to `x`)
    let input = ag::Feed::new().add(x, ndarray::arr1(&[2.]));
    assert_eq!(8., g2.eval_with_input(input)[0]);

    // ddz/dx
    assert_eq!(4., gg.eval()[0]);
}
