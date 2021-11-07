use autograd as ag;

fn large_graph() {
    ag::run(|g| {
        let x: ag::Tensor<f32> = g.ones(&[6]);
        let y: ag::Tensor<f32> = g.ones(&[6]);
        let mut k = x + y;
        let z = g.reshape(x, &[2, 3]);
        for _ in 0..4000 {
            k = k + y;
        }
        let _ = g.eval(&[x, k, z], &[]);
    });
}

pub fn main() {
    large_graph()
}
