extern crate autograd as ag;
extern crate ndarray;


#[test]
fn scalar_add()
{
    let mut graph = ag::Graph::new();
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = 3. + ones + 2;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[6., 6., 6.]).into_dyn());
}

#[test]
fn scalar_sub()
{
    let mut graph = ag::Graph::new();
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = 3. - ones - 2;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[0., 0., 0.]).into_dyn());
}

#[test]
fn scalar_mul()
{
    let mut graph = ag::Graph::new();
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = 3. * ones * 2;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[6., 6., 6.]).into_dyn());
}

#[test]
fn scalar_div()
{
    let mut graph = ag::Graph::new();
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = 3. / ones / 2;
    assert_eq!(
        graph.eval(&[z])[0],
        ndarray::arr1(&[1.5, 1.5, 1.5]).into_dyn()
    );
}

#[test]
fn add()
{
    let mut graph = ag::Graph::new();
    let ref zeros = graph.constant(ndarray::arr1(&[0., 0., 0.]));
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = zeros + ones;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[1., 1., 1.]).into_dyn());
}

#[test]
fn sub()
{
    let mut graph = ag::Graph::new();
    let ref zeros = graph.constant(ndarray::arr1(&[0., 0., 0.]));
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = ones - zeros;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[1., 1., 1.]).into_dyn());
}

#[test]
fn mul()
{
    let mut graph = ag::Graph::new();
    let ref zeros = graph.constant(ndarray::arr1(&[0., 0., 0.]));
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = zeros * ones;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[0., 0., 0.]).into_dyn());
}

#[test]
fn div()
{
    let mut graph = ag::Graph::new();
    let ref zeros = graph.constant(ndarray::arr1(&[0., 0., 0.]));
    let ref ones = graph.constant(ndarray::arr1(&[1., 1., 1.]));
    let ref z: ag::Tensor = zeros / ones;
    assert_eq!(graph.eval(&[z])[0], ndarray::arr1(&[0., 0., 0.]).into_dyn());
}
