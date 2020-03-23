extern crate autograd as ag;
extern crate ndarray;

#[test]
fn random_uniform() {
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = rng.random_uniform(&[3], 0., 1.);
    let b = rng.random_uniform(&[3], 0., 1.);
    assert!(a.all_close(&ndarray::arr1(&[0.5, 0.5, 0.5]), 0.5));
    assert_ne!(a, b);
}

#[test]
fn random_normal() {
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = rng.random_normal(&[3], 0., 1.);
    let b = rng.random_normal(&[3], 0., 1.);
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn standard_normal() {
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = rng.standard_normal(&[3]);
    let b = rng.standard_normal(&[3]);
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn standard_uniform() {
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = rng.standard_uniform(&[3]);
    let b = rng.standard_uniform(&[3]);
    assert!(a.all_close(&ndarray::arr1(&[0.5, 0.5, 0.5]), 0.5));
    assert_ne!(a, b);
}

#[test]
fn glorot_normal() {
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = rng.glorot_normal(&[3, 2]);
    let b = rng.glorot_normal(&[3, 2]);
    assert_ne!(a, b);
}

#[test]
fn glorot_uniform() {
    let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
    let a = rng.glorot_uniform(&[3, 2]);
    let b = rng.glorot_uniform(&[3, 2]);
    assert_ne!(a, b);
}
