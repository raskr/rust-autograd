extern crate autograd as ag;
extern crate ndarray;

use ag::tensor_ops as T;

struct MultiOutputOp;

impl ag::op::Op<f32> for MultiOutputOp {
    fn name(&self) -> &'static str {
        "MultiOutputOp"
    }

    fn compute(&self, ctx: &mut ag::op::ComputeContext<f32>) -> Result<(), ag::op::OpError> {
        let a = ag::ndarray_ext::zeros(&[2, 3]);
        let b = ag::ndarray_ext::zeros(&[1, 3]);
        ctx.append_output(a);
        ctx.append_output(b);
        Ok(())
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f32>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}

#[test]
fn test_nth_tensor() {
    let ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let a = ag::Tensor::builder(g).build(MultiOutputOp);
        let b = T::nth_tensor(a, 1);
        let c = T::exp(b);
        c.eval(g).unwrap();
    });
}

#[test]
fn test_hook() {
    let ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        let a: ag::Tensor<f64> = T::ones(&[4, 2], g).show();
        let b: ag::Tensor<f64> = T::zeros(&[2, 3], g).show_shape();
        let c = T::matmul(a, b).print("aaa");
        c.eval(g).unwrap();
    });
    ctx.run(|g| {
        let x = g.placeholder("x", &[]);
        let y = g.placeholder("x", &[]);
        let z = 2. * x * x + 3. * y + 1.;

        // dz/dy
        let gy = &T::grad(&[z], &[y])[0];
        println!("{:?}", gy.eval(g)); // => Some(3.)

        // dz/dx (requires to fill the placeholder `x`)
        let gx = &T::grad(&[z], &[x])[0];
        let gx_value = g
            .evaluator()
            .push(gx)
            .feed(x, ag::ndarray::arr0(2.).view())
            .run();
        println!("{:?}", gx_value[0]); // => Some(8.)

        // ddz/dx (differentiates `z` again)
        let ggx = &T::grad(&[gx], &[x])[0];
        println!("{:?}", ggx.eval(g)); // => Some(4.)
    });
}

#[test]
#[should_panic]
fn test_too_many_nodes() {
    let ctx = ag::VariableEnvironment::new();
    ctx.run(|g| {
        for _ in 0..10000 {
            let x = g.placeholder("x", &[3]);
            let z = 2.0 * x / 2.0 / 2.0;
            T::grad(&[z], &[x])[0];
        }
    });
}
