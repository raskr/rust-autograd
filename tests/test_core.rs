extern crate autograd as ag;
extern crate ndarray;

struct MultiOutputOp;

impl ag::op::Op<f32> for MultiOutputOp {
    fn name(&self) -> &str {
        "MultiOutputOp"
    }

    fn compute(&self, ctx: &mut ag::op::ComputeContext<f32>) {
        let a = ag::ndarray_ext::zeros(&[2, 3]);
        let b = ag::ndarray_ext::zeros(&[1, 3]);
        ctx.append_output(a);
        ctx.append_output(b);
    }

    fn grad(&self, ctx: &mut ag::op::GradientContext<f32>) {
        ctx.append_input_grad(None);
        ctx.append_input_grad(None);
    }
}

#[test]
fn test_nth_tensor() {
    ag::with(|g| {
        let a = ag::Tensor::builder().build(g, MultiOutputOp);
        let b = g.nth_tensor(a, 1);
        let c = g.exp(b);
        g.eval(&[c], &[]);
    });
}

#[test]
fn test_hook() {
    ag::with(|g| {
        let a: ag::Tensor<f32> = g.ones(&[4, 2]).show();
        let b: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape();
        let c = g.matmul(a, b).print("aaa");
        g.eval(&[c], &[]);
    });
    ag::with(|g: &mut ag::Graph<_>| {
        let x = g.placeholder(&[]);
        let y = g.placeholder(&[]);
        let z = 2. * x * x + 3. * y + 1.;

        // dz/dy
        let gy = &g.grad(&[z], &[y])[0];
        println!("{:?}", gy.eval(&[])); // => Some(3.)

        // dz/dx (requires to fill the placeholder `x`)
        let gx = &g.grad(&[z], &[x])[0];
        println!("{:?}", gx.eval(&[x.given(ag::ndarray::arr0(2.).view())])); // => Some(8.)

        // ddz/dx (differentiates `z` again)
        let ggx = &g.grad(&[gx], &[x])[0];
        println!("{:?}", ggx.eval(&[])); // => Some(4.)
    });
}

#[test]
fn test_many_nodes() {
    ag::with(|g: &mut ag::Graph<f64>| {
        for _ in 0..10000 {
            let x = g.placeholder(&[3]);
            let z = 2.0 * x / 2.0 / 2.0;
            g.grad(&[z], &[x])[0];
        }
    });
}
