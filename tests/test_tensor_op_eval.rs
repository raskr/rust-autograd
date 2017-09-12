extern crate autograd as ag;
extern crate ndarray;
use self::ndarray::arr2;


#[test]
fn tile() {
    let ref x = ag::constant(ndarray::arr2(&[[2., 2.], [3., 3.]]));
    let ref y = ag::tile(x, 0, 2);
    assert_eq!(
        y.eval(),
        ndarray::arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]).into_dyn()
    );
}

#[test]
fn clip() {
    let ref x = ag::constant(ndarray::arr1(&[2., 4., 6.]));
    let ref y = ag::clip(x, 3., 5.);
    assert_eq!(y.eval(), ndarray::arr1(&[3., 4., 5.]).into_dyn());
}

#[test]
fn reduce_max() {
    let x = ag::constant(ndarray::arr2(&[[2.], [4.], [6.]]));
    let y = ag::reduce_max(&x, 0, false);
    assert_eq!(y.eval()[0], 6.);
}

#[test]
fn reduce_mean() {
    let x = ag::constant(ndarray::arr2(&[[2.], [4.], [6.]]));
    let y = ag::reduce_mean(&x, 0, false);
    assert_eq!(y.eval()[0], 4.);
}

#[test]
fn argmax() {
    let input_arr = ndarray::arr2(&[[1., 2.], [3., 4.], [6., 5.]]);
    let answer = ndarray::arr1(&[1., 1., 0.]).into_dyn();
    let input = ag::constant(input_arr);
    let result = ag::argmax(&input, 1, false);
    assert_eq!(result.eval(), answer);
}

#[test]
fn argmax_keep() {
    let input_arr = ndarray::arr2(&[[1., 2.], [3., 4.], [6., 5.]]);
    let answer = ndarray::arr2(&[[1.], [1.], [0.]]).into_dyn();
    let input = ag::constant(input_arr);
    let result = ag::argmax(&input, 1, true);
    assert_eq!(result.eval(), answer);
}

#[test]
fn reshape() {
    let input_arr = ag::init::standard_normal(&[3, 2, 2]);
    let answer = input_arr
        .clone()
        .into_shape(ndarray::IxDyn(&[3, 4]))
        .unwrap();
    let x = ag::constant(input_arr);
    let y = ag::reshape(&x, &[3, 4]);
    assert_eq!(y.eval(), answer);
}
