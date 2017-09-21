extern crate autograd as ag;
extern crate ndarray;

use std::mem;


#[test]
fn variable()
{
    let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]));
    let v = ag::variable(arr.clone());
    let from_variable = mem::replace(&mut v.borrow_mut().param, None).unwrap();
    assert!(arr == from_variable);
}

#[test]
fn constant()
{
    let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[3]));
    let v = ag::constant(arr.clone());
    let from_variable = mem::replace(&mut v.borrow_mut().param, None).unwrap();
    assert!(arr == from_variable);
}

#[test]
fn placeholder()
{
    let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[2]));
    let p = ag::placeholder(&[2]);
    assert_eq!(
        arr,
        p.eval_with_input(ag::Feed::new().add(&p, arr.clone()))
    )
}
