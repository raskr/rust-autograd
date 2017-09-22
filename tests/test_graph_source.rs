extern crate autograd as ag;
extern crate ndarray;

use std::mem;


#[test]
fn variable()
{
    let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let v = ag::variable(arr.clone());
    let from_variable = mem::replace(&mut v.borrow_mut().param, None).unwrap();
    assert!(arr == from_variable);
}

#[test]
fn constant()
{
    let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let v = ag::constant(arr.clone());
    let from_variable = mem::replace(&mut v.borrow_mut().param, None).unwrap();
    assert!(arr == from_variable);
}

#[test]
fn placeholder()
{
    let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let p = ag::placeholder();
    assert_eq!(arr, p.eval_with_input(ag::Feed::new().add(&p, arr.clone())))
}
