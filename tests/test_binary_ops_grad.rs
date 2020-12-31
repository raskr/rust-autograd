extern crate autograd as ag;
extern crate ndarray;

use ag::tensor::Variable;
use ag::with;
use ndarray::array;

#[test]
fn scalar_add() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = x + 2.;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(1., grad.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_sub() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = x - 2.;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(1., grad.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_mul() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = 3. * x;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(3., grad.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_div() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = x / 3.;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(1. / 3., grad.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr1() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = 3. * x + 2.;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(3., grad.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr2() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = 3. * x * x;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(
            18.,
            grad.eval(&[x.given(ndarray::arr0(3.).view())]).unwrap()[ndarray::IxDyn(&[])]
        );
    });
}

#[test]
fn expr3() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = 3. * x * x + 2.;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(
            18.,
            grad.eval(&[x.given(ndarray::arr0(3.).view())]).unwrap()[ndarray::IxDyn(&[])]
        );
    });
}

#[test]
fn expr4() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = 3. * x * x + 2. * x + 1.;
        let grad = g.grad(&[y], &[x])[0];
        assert_eq!(
            20.,
            grad.eval(&[x.given(ndarray::arr0(3.).view())]).unwrap()[ndarray::IxDyn(&[])]
        );
    });
}

#[test]
fn expr5() {
    ag::with(|g| {
        let x1 = g.placeholder(&[]);
        let x2 = g.placeholder(&[]);
        let y = 3. * x1 * x1 + 2. * x1 + x2 + 1.;
        let grad = g.grad(&[y], &[x1])[0];
        assert_eq!(
            20.,
            grad.eval(&[x1.given(ndarray::arr0(3.).view())]).unwrap()[ndarray::IxDyn(&[])]
        );
    });
}

#[test]
// Test with intention that grad of `x2` should be computed
// even if the value of `x1` is not given
fn expr6() {
    ag::with(|g| {
        let x1 = g.placeholder(&[]);
        let x2 = g.variable(ndarray::arr0(0.));
        let y = 3. * x1 * x1 + 5. * x2;
        let grad = g.grad(&[y], &[x2])[0];
        assert_eq!(5., grad.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn differentiate_twice() {
    ag::with(|g| {
        let x = g.placeholder(&[]);
        let y = x * x;
        let g1 = g.grad(&[y], &[x])[0];
        let g2 = g.grad(&[g1], &[x])[0];
        assert_eq!(2., g2.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr7() {
    ag::with(|g| {
        let x1 = g.placeholder(&[]);
        let x2 = g.placeholder(&[]);
        let y = 2. * x1 * x1 + 3. * x2;
        let g1 = g.grad(&[y], &[x1])[0];
        let g2 = g.grad(&[y], &[x2])[0];
        let gg1 = g.grad(&[g1], &[x1])[0];
        assert_eq!(3., g2.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
        assert_eq!(4., gg1.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
        assert_eq!(
            8.,
            g1.eval(&[x1.given(ndarray::arr0(2.).view())]).unwrap()[ndarray::IxDyn(&[])]
        );
    });
}

#[test]
fn expr8() {
    ag::with(|graph| {
        let x1 = graph.placeholder(&[]);
        let y = x1 * x1 * x1 * x1;
        let g = graph.grad(&[y], &[x1])[0];
        let gg = graph.grad(&[g], &[x1])[0];
        let ggg = graph.grad(&[gg], &[x1])[0];
        assert_eq!(
            48.,
            ggg.eval(&[x1.given(ndarray::arr0(2.).view())]).unwrap()[ndarray::IxDyn(&[])]
        );
    });
}

#[test]
fn scalar_tensor_add() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v2 + v;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_add2() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v + v2;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_sub() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v2 - v;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_sub2() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v - v2;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_mul() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v2 * v;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_mul2() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v * v2;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_div() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v / v2;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_div2() {
    with(|graph| {
        let v = graph.variable(ndarray::arr0(1.));
        let v2 = graph.variable(ndarray::arr1(&[1., 2., 3.]));
        let a: ag::Tensor<f64> = v2 / v;
        let g = graph.grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(a, g.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_mul_g3() {
    ag::with(|graph| {
        let v = graph.variable(ag::ndarray_ext::from_scalar(2.));
        let three = graph.variable(array![3., 3., 3.]);
        let y = three * v * v * v;
        let g = graph.grad(&[y], &[v])[0];
        let gg = graph.grad(&[g], &[v])[0];
        let ggg = graph.grad(&[gg], &[v]);
        assert_eq!(
            (&ggg[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(gg, ggg.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}

#[test]
fn scalar_tensor_div_g3() {
    ag::with(|graph| {
        let v = graph.variable(ag::ndarray_ext::from_scalar(2.));
        let three = graph.variable(array![3., 3., 3.]);
        let y = three / v / v / v;
        let g = graph.grad(&[y], &[v])[0];
        let gg = graph.grad(&[g], &[v])[0];
        let ggg = graph.grad(&[gg], &[v]);
        assert_eq!(
            (&ggg[0]).eval(&[]).unwrap().shape(),
            v.eval(&[]).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(gg, ggg.as_slice(), &[v], &[], 1e-3, 1e-3);
    });
}
