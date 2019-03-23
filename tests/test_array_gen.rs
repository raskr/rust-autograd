extern crate autograd as ag;
extern crate ndarray;

#[test]
fn random_uniform() {
    let a = ag::ndarray_ext::random_uniform(&[3], 0., 1.);
    let b = ag::ndarray_ext::random_uniform(&[3], 0., 1.);
    assert!(a.all_close(&ndarray::arr1(&[0.5, 0.5, 0.5]), 0.5));
    assert_ne!(a, b);
}

#[test]
fn random_normal() {
    let a = ag::ndarray_ext::random_normal::<f32>(&[3], 0., 1.);
    let b = ag::ndarray_ext::random_normal(&[3], 0., 1.);
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn standard_normal() {
    let a = ag::ndarray_ext::standard_normal::<f32>(&[3]);
    let b = ag::ndarray_ext::standard_normal(&[3]);
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn standard_uniform() {
    let a = ag::ndarray_ext::standard_uniform::<f32>(&[3]);
    let b = ag::ndarray_ext::standard_uniform(&[3]);
    assert!(a.all_close(&ndarray::arr1(&[0.5, 0.5, 0.5]), 0.5));
    assert_ne!(a, b);
}

#[test]
fn glorot_normal() {
    let a = ag::ndarray_ext::glorot_normal::<f32>(&[3, 2]);
    let b = ag::ndarray_ext::glorot_normal(&[3, 2]);
    assert_ne!(a, b);
}

#[test]
fn glorot_uniform() {
    let a = ag::ndarray_ext::glorot_uniform::<f32>(&[3, 2]);
    let b = ag::ndarray_ext::glorot_uniform(&[3, 2]);
    assert_ne!(a, b);
}
