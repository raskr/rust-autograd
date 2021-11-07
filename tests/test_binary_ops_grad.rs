extern crate autograd as ag;
extern crate ndarray;

use ag::prelude::*;
use ag::tensor_ops as T;
use ag::VariableEnvironment;
use ndarray::array;

#[test]
fn scalar_add() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = x + 2.;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(1., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_sub() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = x - 2.;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(1., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_mul() {
    let mut ctx = ag::VariableEnvironment::<f64>::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = 3. * x;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(3., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_div() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = x / 3.;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(1. / 3., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_div2() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = x / 3. / 4.;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(1. / 3. / 4., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr0() {
    let mut ctx = ag::VariableEnvironment::<f64>::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = 3. * x;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(3., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr1() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = 3. * x + 2.;
        let grad = T::grad(&[y], &[x])[0];
        assert_eq!(3., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr2() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = 3. * x * x;
        let grad = T::grad(&[y], &[x])[0];
        let eval = g
            .evaluator()
            .push(grad)
            .feed(x, ndarray::arr0(3.).view())
            .run();
        assert_eq!(18., eval[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr3() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = 3. * x * x + 2.;
        let grad = T::grad(&[y], &[x])[0];
        let eval = g
            .evaluator()
            .push(grad)
            .feed(x, ndarray::arr0(3.).view())
            .run();
        assert_eq!(18., eval[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr4() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = 3. * x * x + 2. * x + 1.;
        let grad = T::grad(&[y], &[x])[0];
        let eval = g
            .evaluator()
            .feed(x, ndarray::arr0(3.).view())
            .push(grad)
            .run();
        assert_eq!(20., eval[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr5() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x1 = g.placeholder("x1", &[]);
        let x2 = g.placeholder("x2", &[]);
        let y = 3. * x1 * x1 + 2. * x1 + x2 + 1.;
        let grad = T::grad(&[y], &[x1])[0];
        let eval = g
            .evaluator()
            .push(grad)
            .feed(x1, ndarray::arr0(3.).view())
            .run();
        assert_eq!(20., eval[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
// Test with intention that grad of `x2` should be computed
// even if the value of `x1` is not given
fn expr6() {
    let mut env = VariableEnvironment::new();
    let x2 = env.slot().set(ndarray::arr0(0.));
    env.run(|g| {
        let x1 = g.placeholder("x1", &[]);
        let x2 = g.variable(x2);
        let y = 3. * x1 * x1 + 5. * x2;
        let grad = T::grad(&[y], &[x2])[0];
        assert_eq!(5., grad.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn differentiate_twice() {
    let mut ctx = ag::VariableEnvironment::<f64>::new();
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = x * x;
        let g1 = T::grad(&[y], &[x])[0];
        let g2 = T::grad(&[g1], &[x])[0];
        assert_eq!(2., g2.eval(g).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr7() {
    let mut ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let x1 = g.placeholder("x1", &[]);
        let x2 = g.placeholder("x2", &[]);
        let y = 2. * x1 * x1 + 3. * x2;
        let g1 = T::grad(&[y], &[x1])[0];
        let g2 = T::grad(&[y], &[x2])[0];
        let gg1 = T::grad(&[g1], &[x1])[0];
        assert_eq!(3., g2.eval(g).unwrap()[ndarray::IxDyn(&[])]);
        assert_eq!(4., gg1.eval(g).unwrap()[ndarray::IxDyn(&[])]);
        let mut eval = g
            .evaluator()
            .push(g1)
            .feed(x1, ndarray::arr0(2.).view())
            .run();
        assert_eq!(8., eval[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn expr8() {
    let mut env = ag::VariableEnvironment::new();
    env.run(|graph| {
        let x1 = graph.placeholder("x1", &[]);
        let y = x1 * x1 * x1 * x1;
        let g = T::grad(&[y], &[x1])[0];
        let gg = T::grad(&[g], &[x1])[0];
        let ggg = T::grad(&[gg], &[x1])[0];
        let eval = graph
            .evaluator()
            .push(ggg)
            .feed(x1, ndarray::arr0(2.).view())
            .run();

        assert_eq!(48., eval[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn scalar_tensor_add() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v2 + v;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_add2() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v + v2;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_sub() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v2 - v;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_sub2() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v - v2;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_mul() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v2 * v;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_mul2() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v * v2;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_div() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v / v2;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_div2() {
    let mut env = ag::VariableEnvironment::new();
    let v = env.slot().set(ndarray::arr0(1.));
    let v2 = env.slot().set(ndarray::arr1(&[1., 2., 3.]));
    env.run(|graph| {
        let v = graph.variable(v);
        let v2 = graph.variable(v2);
        let a: ag::Tensor<f64> = v2 / v;
        let g = T::grad(&[a], &[v]);
        assert_eq!(
            (&g[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            a,
            g.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_mul_g3() {
    let mut env = ag::VariableEnvironment::new();
    let three = env.slot().set(array![3., 3., 3.]);
    let v = env.slot().set(ag::ndarray_ext::from_scalar(2.));
    env.run(|graph| {
        let v = graph.variable(v);
        let three = graph.variable(three);
        let y = three * v * v * v;
        let g = T::grad(&[y], &[v])[0];
        let gg = T::grad(&[g], &[v])[0];
        let ggg = T::grad(&[gg], &[v]);
        assert_eq!(
            (&ggg[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            gg,
            ggg.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}

#[test]
fn scalar_tensor_div_g3() {
    let mut env = ag::VariableEnvironment::new();
    let three = env.slot().set(array![3., 3., 3.]);
    let v = env.slot().set(ag::ndarray_ext::from_scalar(2.));
    env.run(|graph| {
        let v = graph.variable(v);
        let three = graph.variable(three);
        let y = three / v / v / v;
        let g = T::grad(&[y], &[v])[0];
        let gg = T::grad(&[g], &[v])[0];
        let ggg = T::grad(&[gg], &[v]);
        assert_eq!(
            (&ggg[0]).eval(graph).unwrap().shape(),
            v.eval(graph).unwrap().shape()
        );
        ag::test_helper::check_theoretical_grads(
            gg,
            ggg.as_slice(),
            &[v],
            ag::Feeder::new(),
            1e-3,
            1e-3,
            graph,
        );
    });
}
