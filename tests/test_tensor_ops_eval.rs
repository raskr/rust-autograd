extern crate autograd as ag;
extern crate ndarray;

#[test]
fn reduce_prod() {
    let ref v = ag::variable(ag::ndarray_ext::standard_normal::<f32>(&[3, 2]));
    let ref z = ag::reduce_prod(v, &[0, 1], false); // keep_dims=false
    let empty_shape: &[usize] = &[];
    assert_eq!(z.eval(&[]).unwrap().shape(), empty_shape);
}

#[test]
fn argmax() {
    let ref x = ag::constant(ndarray::arr2(&[[3., 4.], [5., 6.]]));
    let ref y = ag::argmax(x, -1, false);
    assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[1., 1.]).into_dyn()));
}

#[test]
fn argmax_with_multi_max_args() {
    let ref x = ag::constant(ndarray::arr1(&[1., 2., 3., 3.]));
    let ref y = ag::argmax(x, 0, false);
    assert_eq!(2., y.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
}

#[test]
fn reduce_mean() {
    let ref v = ag::variable(ndarray::arr1(&[2., 3., 4.]));
    let ref z = ag::reduce_mean(v, &[0], false); // keep_dims=false
    assert_eq!(3., z.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
}

#[test]
fn reduce_grad() {
    let ref v = ag::variable(ndarray::arr1(&[2., 3., 4.]));
    let ref z = ag::reduce_mean(v, &[0], false); // keep_dims=false
    let ref g = ag::grad(&[z], &[v])[0];
    assert_eq!(g.eval(&[]).unwrap().shape(), &[3]);
}
