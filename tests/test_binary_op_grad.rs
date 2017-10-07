extern crate autograd as ag;
extern crate ndarray;



#[test]
fn scalar_add()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.variable(ndarray::arr1(&[]));
    let ref y = x + 2;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    assert_eq!(1., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn scalar_sub()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.variable(ndarray::arr1(&[]));
    let ref y = x - 2;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    assert_eq!(1., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn scalar_mul()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.variable(ndarray::arr1(&[]));
    let ref y = 3 * x;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    assert_eq!(3., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn scalar_div()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.variable(ndarray::arr1(&[]));
    let ref y = x / 3;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    assert_eq!(1. / 3., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn expr1()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.variable(ndarray::arr1(&[]));
    let ref y = 3 * x + 2;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    assert_eq!(3., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn expr2()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.placeholder();
    let ref y = 3 * x * x;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    graph.feed(x, ndarray::arr1(&[3.]));
    assert_eq!(18., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn expr3()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.placeholder();
    let ref y = 3 * x * x + 2;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    graph.feed(x, ndarray::arr1(&[3.]));
    assert_eq!(18., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn expr4()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.placeholder();
    let ref y = 3 * x * x + 2 * x + 1;
    let grads = ag::gradients(&[y], &[x], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    graph.feed(x, ndarray::arr1(&[3.]));
    assert_eq!(20., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn expr5()
{
    let mut graph = ag::Graph::new();
    let ref x1 = graph.placeholder();
    let ref x2 = graph.placeholder();
    let ref y = 3 * x1 * x1 + 2 * x1 + x2 + 1;
    let grads = ag::gradients(&[y], &[x1], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    graph.feed(x1, ndarray::arr1(&[3.]));
    assert_eq!(20., graph.eval(grads.as_slice())[0][0]);
}

#[test]
// Test with intention that grad of `x2` should be computed
// even if the value of `x1` is not given
fn expr6()
{
    let mut graph = ag::Graph::new();
    let ref x1 = graph.placeholder();
    let ref x2 = graph.variable(ndarray::arr1(&[]));
    let ref y = 3 * x1 * x1 + 5 * x2 + 1;
    let grads = ag::gradients(&[y], &[x2], &[None]);
    let grads = grads.iter().map(|a| a).collect::<Vec<_>>();
    assert_eq!(5., graph.eval(grads.as_slice())[0][0]);
}

#[test]
fn differentiate_twice()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.placeholder();
    let ref y = x * x;
    let ref g1 = ag::gradients(&[y], &[x], &[None])[0];
    let ref g2 = ag::gradients(&[g1], &[x], &[None])[0];
    graph.feed(x, ndarray::arr1(&[2.]));
    assert_eq!(2., graph.eval(&[g2])[0][0]);
}


#[test]
fn expr7()
{
    let mut graph = ag::Graph::new();
    let ref x1 = graph.placeholder();
    let ref x2 = graph.variable(ag::ndarray_ext::zeros(&[1]));
    let ref y = 2 * x1 * x1 + 3 * x2 + 1;
    let ref g1 = ag::gradients(&[y], &[x1], &[None])[0];
    let ref g2 = ag::gradients(&[y], &[x2], &[None])[0];
    let ref gg1 = ag::gradients(&[g1], &[x1], &[None])[0];

    graph.feed(x1, ndarray::arr1(&[2.]));
    assert_eq!(8., graph.eval(&[g1])[0][0]);
    assert_eq!(3., graph.eval(&[g2])[0][0]);
    assert_eq!(4., graph.eval(&[gg1])[0][0]);
}

#[test]
fn expr8()
{
    let mut graph = ag::Graph::new();
    let ref x = graph.placeholder();
    let ref y = graph.variable(ag::ndarray_ext::zeros(&[1]));
    let ref z = 2 * x * x + 3 * y + 1;
    let ref g1 = ag::gradients(&[z], &[y], &[None])[0];
    let ref g2 = ag::gradients(&[z], &[x], &[None])[0];
    let ref gg = ag::gradients(&[g2], &[x], &[None])[0];

    // dz/dy
    assert_eq!(3., graph.eval(&[g1])[0][0]);

    // dz/dx (necessary to feed the value to `x`)
    graph.feed(x, ndarray::arr1(&[2.]));
    assert_eq!(8., graph.eval(&[g2])[0][0]);

    // ddz/dx
    assert_eq!(4., graph.eval(&[gg])[0][0]);
}
