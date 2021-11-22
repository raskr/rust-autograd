extern crate autograd as ag;
extern crate ndarray;
use self::ag::NdArray;
use ag::prelude::*;
use ag::tensor_ops as T;
use ag::VariableEnvironment;
use ndarray::array;

#[test]
fn reduce_prod() {
    ag::run(|g| {
        let rng = ag::ndarray_ext::ArrayRng::<f64>::default();
        let v = T::convert_to_tensor(rng.standard_normal(&[3, 2]), g);
        let z = T::reduce_prod(v, &[0, 1], false); // keep_dims=false
        let empty_shape: &[usize] = &[];
        assert_eq!(z.eval(g).unwrap().shape(), empty_shape);
    });
}

#[test]
fn argmax() {
    ag::run(|g| {
        let x = T::convert_to_tensor(array![[3., 4.], [5., 6.]], g);
        let y = T::argmax(x, -1, false);
        assert_eq!(y.eval(g), Ok(ndarray::arr1(&[1., 1.]).into_dyn()));
    });
}

#[test]
fn argmax2() {
    ag::run(|g| {
        let test = T::convert_to_tensor(array![84.0, 16.0, 0.04, 85.0, 16.0, 85.0], g);
        let max_dis_index = T::argmax(test, 0, false);
        assert_eq!(max_dis_index.eval(g), Ok(ndarray::arr0(3.).into_dyn()));
    });
}

#[test]
fn argmax_with_multi_max_args() {
    ag::run(|g| {
        let x = T::convert_to_tensor(array![1., 2., 3., 3.], g);
        let y = T::argmax(x, 0, false);
        assert_eq!(2., y.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_mean() {
    let mut env = VariableEnvironment::new();
    let v = env.slot().set(array![2., 3., 4.]);
    env.run(|g| {
        let v = g.variable(v);
        let z = T::reduce_mean(v, &[0], false); // keep_dims=false
        assert_eq!(3., z.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_grad() {
    let mut env = VariableEnvironment::new();
    let v = env.slot().set(array![2., 3., 4.]);
    env.run(|g| {
        let v = g.variable(v);
        let z = T::reduce_mean(v, &[0], false); // keep_dims=false
        let grad = T::grad(&[z], &[v])[0];
        assert_eq!(grad.eval(g).unwrap().shape(), &[3]);
    });
}

#[test]
fn transpose_matmul_square() {
    ag::run(|g| {
        let x = T::convert_to_tensor(array![[0., 1.], [2., 3.]], g);
        let w = T::convert_to_tensor(array![[0., 1.], [2., 3.]], g);
        let w2 = T::transpose(w, &[1, 0]);
        let mm = T::matmul(x, w2);
        assert_eq!(mm.eval(g).unwrap().as_slice().unwrap(), &[1., 3., 3., 13.]);
    });
}

#[test]
fn transpose_matmul() {
    ag::run(|g| {
        let x = T::convert_to_tensor(array![[0., 1., 2.], [3., 4., 5.]], g);
        let w = T::convert_to_tensor(array![[0., 1.], [2., 3.]], g);
        let x2 = T::transpose(x, &[1, 0]).show();
        let mm = T::matmul(x2, w).show();
        assert_eq!(
            mm.eval(g).unwrap().as_slice().unwrap(),
            &[6., 9., 8., 13., 10., 17.]
        );
    });
}

#[test]
fn test_mm() {
    ag::run(|g: &mut ag::Context<f32>| {
        let a = T::ones(&[2, 5], g);
        let b = T::ones(&[5, 1], g);
        let c = T::matmul(&a, &b);
        let d = c.eval(g).unwrap();
        assert_eq!(d.as_slice().unwrap(), &[5., 5.]);
    });
}

#[test]
fn test_batch_matmul_normal() {
    // blas is used
    ag::run(|g: &mut ag::Context<f32>| {
        let a: ag::Tensor<f32> = T::ones(&[2, 3, 4, 2], g);
        let b: ag::Tensor<f32> = T::ones(&[2, 3, 2, 3], g);
        let c = T::batch_matmul(a, b);
        let shape = &[2, 3, 4, 3];
        let size = shape.iter().product();
        let ans = NdArray::<_>::from_shape_vec(ndarray::IxDyn(shape), vec![2f32; size]).unwrap();
        let ret = c.eval(g).unwrap();
        ret.all_close(&ans, 1e-4);
    });
}

#[test]
fn test_batch_matmul_trans_not_square() {
    ag::run(|g: &mut ag::Context<f32>| {
        let a: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let b: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let c = T::batch_matmul(a, b);
        let ans = array![[[7., 10.], [15., 22.]], [[7., 10.], [15., 22.]]].into_dyn();
        let ret = c.eval(g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_trans_square_both() {
    // blas is not used
    ag::run(|g: &mut ag::Context<f32>| {
        let a_: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let b_: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let a: ag::Tensor<f32> = T::transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> = T::transpose(b_, &[0, 2, 1]);
        let c = T::batch_matmul(a, b);
        let ans = array![[[7., 15.], [10., 22.]], [[7., 15.], [10., 22.]]].into_dyn();
        let ret = c.eval(g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_trans_square_lhs() {
    // blas is used
    ag::run(|g: &mut ag::Context<f32>| {
        let a_: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let a: ag::Tensor<f32> = T::transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let c = T::batch_matmul(a, b);
        let ans = array![[[10., 14.], [14., 20.]], [[10., 14.], [14., 20.]]].into_dyn();
        let ret = c.eval(g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}

#[test]
fn test_batch_matmul_with_copy() {
    ag::run(|g: &mut ag::Context<f32>| {
        let a_: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let a: ag::Tensor<f32> = T::transpose(a_, &[0, 2, 1]);
        let b: ag::Tensor<f32> =
            T::convert_to_tensor(array![[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]], g);
        let c = T::batch_matmul(a, b);
        let ans = array![[[10., 14.], [14., 20.]], [[10., 14.], [14., 20.]]].into_dyn();
        let ret = c.eval(g).unwrap();
        assert!(ret.all_close(&ans, 1e-4));
        assert_eq!(ret.shape(), &[2, 2, 2]);
    });
}
