extern crate autograd as ag;
extern crate ndarray;
use self::ag::NdArray;
use ag::{tensor::Constant, tensor::Variable, with};
use ndarray::array;

#[test]
fn reduce_prod() {
    with(|g| {
        let v = g.constant(ag::ndarray_ext::standard_normal::<f32>(&[3, 2]));
        let z = g.reduce_prod(v, &[0, 1], false); // keep_dims=false
        let empty_shape: &[usize] = &[];
        assert_eq!(z.eval(&[]).unwrap().shape(), empty_shape);
    });
}

#[test]
fn argmax() {
    with(|g| {
        let x = g.constant(array![[3., 4.], [5., 6.]]);
        let y = g.argmax(x, -1, false);
        assert_eq!(y.eval(&[]), Ok(ndarray::arr1(&[1., 1.]).into_dyn()));
    });
}

#[test]
fn argmax_with_multi_max_args() {
    with(|g| {
        let x = g.constant(array![1., 2., 3., 3.]);
        let y = g.argmax(x, 0, false);
        assert_eq!(2., y.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_mean() {
    with(|g| {
        let v = g.variable(array![2., 3., 4.]);
        let z = g.reduce_mean(v, &[0], false); // keep_dims=false
        assert_eq!(3., z.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_grad() {
    with(|g| {
        let v = g.variable(array![2., 3., 4.]);
        let z = g.reduce_mean(v, &[0], false); // keep_dims=false
        let g = g.grad(&[z], &[v])[0];
        assert_eq!(g.eval(&[]).unwrap().shape(), &[3]);
    });
}

#[test]
fn transpose_matmul_square() {
    with(|g| {
        let x = g.constant(array![[0., 1.], [2., 3.]]);
        let w = g.constant(array![[0., 1.], [2., 3.]]);
        let w2 = g.transpose(w, &[1, 0]);
        let mm = g.matmul(x, w2);
        assert_eq!(
            mm.eval(&[]).unwrap().as_slice().unwrap(),
            &[1., 3., 3., 13.]
        );
    });
}

#[test]
fn transpose_matmul() {
    with(|g| {
        let x = g.constant(array![[0., 1., 2.], [3., 4., 5.]]);
        let w = g.constant(array![[0., 1.], [2., 3.]]);
        let x2 = g.transpose(x, &[1, 0]).show();
        let mm = g.matmul(x2, w).show();
        assert_eq!(
            mm.eval(&[]).unwrap().as_slice().unwrap(),
            &[6., 9., 8., 13., 10., 17.]
        );
    });
}

#[test]
fn test_mm() {
    with(|g: &mut ag::Graph<f32>| {
        let a = g.ones(&[2, 5]);
        let b = g.ones(&[5, 1]);
        let c = g.matmul(&a, &b);
        let d = c.eval(&[]).unwrap();
        assert_eq!(d.as_slice().unwrap(), &[5., 5.]);
    });
}

#[test]
fn test_batch_matmul_normal() {
    // blas is used
    with(|g: &mut ag::Graph<f32>| {
        let a: ag::Tensor<f32> = g.ones(&[2, 3, 4, 2]);
        let b: ag::Tensor<f32> = g.ones(&[2, 3, 2, 3]);
        let c = g.batch_matmul(a, b);
        let shape = &[2, 3, 4, 3];
        let size = shape.iter().product();
        let ans = NdArray::<_>::from_shape_vec(ndarray::IxDyn(shape), vec![2f32; size]).unwrap();
        let ret = c.eval(&[]).unwrap();
        ret.all_close(&ans, 1e-4);
    });
}

#[test]
fn test_batch_matmul_trans_not_square() {
    // blas is not used
    with(|g: &mut ag::Graph<f32>| {
        let a: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let b: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let c = g.batch_matmul(a, b);
        let ans = array![[[7., 10.], [15., 22.]], [[7., 10.], [15., 22.]]].into_dyn();
        let ret = c.eval(&[]).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_trans_square_both() {
    // blas is not used
    with(|g: &mut ag::Graph<f32>| {
        let a_: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let b_: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let a: ag::Tensor<f32> = g.transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> = g.transpose(b_, &[0, 2, 1]);
        let c = g.batch_matmul(a, b);
        let ans = array![[[7., 15.], [10., 22.]], [[7., 15.], [10., 22.]]].into_dyn();
        let ret = c.eval(&[]).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_trans_square_lhs() {
    // blas is used
    with(|g: &mut ag::Graph<f32>| {
        let a_: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let a: ag::Tensor<f32> = g.transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let c = g.batch_matmul(a, b);
        let ans = array![[[10., 14.], [14., 20.]], [[10., 14.], [14., 20.]]].into_dyn();
        let ret = c.eval(&[]).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_with_copy() {
    // blas is used
    with(|g: &mut ag::Graph<f32>| {
        let a_: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let a: ag::Tensor<f32> = g.transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> = g.constant(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]);
        let c = g.batch_matmul(a, b);
        let ans = array![[[10., 14.], [14., 20.]], [[10., 14.], [14., 20.]]].into_dyn();
        let ret = c.eval(&[]).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}
