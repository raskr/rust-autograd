extern crate autograd as ag;
extern crate ndarray;


#[test]
fn zeros() {
    let a = ag::initializers::zeros(&[3]);
    let b = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]));
    assert_eq!(a, b);
}

#[test]
fn ones() {
    let a = ag::initializers::ones(&[3]);
    let b = ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 1.);
    assert_eq!(a, b);
}

#[test]
fn random_uniform() {
    let a = ag::initializers::random_uniform(&[3], 0., 1.);
    let b = ag::initializers::random_uniform(&[3], 0., 1.);
    assert!(a.all_close(&ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 0.5), 0.5));
    assert_ne!(a, b);
}

#[test]
fn random_normal() {
    let a = ag::initializers::random_normal(&[3], 0., 1.);
    let b = ag::initializers::random_normal(&[3], 0., 1.);
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn standard_normal() {
    let a = ag::initializers::standard_normal(&[3]);
    let b = ag::initializers::standard_normal(&[3]);
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn standard_uniform() {
    let a = ag::initializers::standard_normal(&[3]);
    let b = ag::initializers::standard_normal(&[3]);
    assert!(a.all_close(&ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[3]), 0.5), 0.5));
    assert_ne!(a, b);
    assert_eq!(a.shape(), &[3])
}

#[test]
fn glorot_normal() {
    let a = ag::initializers::glorot_normal(&[3]);
    let b = ag::initializers::glorot_normal(&[3]);
    assert_ne!(a, b);
}

#[test]
fn glorot_uniform() {
    let a = ag::initializers::glorot_uniform(&[3]);
    let b = ag::initializers::glorot_uniform(&[3]);
    assert_ne!(a, b);
}
