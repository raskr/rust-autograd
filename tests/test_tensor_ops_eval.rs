extern crate autograd as ag;
extern crate ndarray;
use self::ndarray::arr2;


#[test]
fn tile() {
    let ref x = ag::constant(arr2(&[[2., 2.], [3., 3.]]).into_dyn());
    let ref y = ag::tile(x, 0, 2);
    assert_eq!(y.eval(), arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]).into_dyn());
}

#[test]
fn clip() {
    let ref x = ag::constant(ag::init::from_slice(&[2., 4., 6.], &[3]));
    let ref y = ag::clip(x, 3., 5.);
    assert_eq!(y.eval(), ag::init::from_slice(&[3., 4., 5.], &[3]));
}

#[test]
fn reduce_max() {
    let x = ag::constant(ag::init::from_slice(&[2., 4., 6.], &[3, 1]));
    let y = ag::reduce_max(&x, 0, false);
    assert_eq!(y.eval()[0], 6.);
}

#[test]
fn reduce_mean() {
    let x = ag::constant(ag::init::from_slice(&[2., 4., 6.], &[3, 1]));
    let y = ag::reduce_mean(&x, 0, false);
    assert_eq!(y.eval()[0], 4.);
}

#[test]
fn reshape() {
    let input_arr = ag::init::randn(&[3, 2, 2]);
    let answer = input_arr
        .clone()
        .into_shape(ndarray::IxDyn(&[3, 4]))
        .unwrap();
    let x = ag::constant(input_arr);
    let y = ag::reshape(&x, &[3, 4]);
    assert_eq!(y.eval(), answer);
}

#[test]
fn argmax() {
    let input_arr = ag::init::from_slice(&[1., 2., 3., 4., 6., 5.], &[3, 2]);
    let answer = ag::init::from_slice(&[1., 1., 0.], &[3]);
    let input = ag::constant(input_arr);
    let result = ag::argmax(&input, 1, false);
    assert_eq!(result.eval(), answer);
}

#[test]
fn argmax_keep() {
    let input_arr = ag::init::from_slice(&[1., 2., 3., 4., 6., 5.], &[3, 2]);
    let answer = ag::init::from_slice(&[1., 1., 0.], &[3, 1]);
    let input = ag::constant(input_arr);
    let result = ag::argmax(&input, 1, true);
    assert_eq!(result.eval(), answer);
}
