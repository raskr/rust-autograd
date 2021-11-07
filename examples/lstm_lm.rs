use autograd as ag;
use ndarray;

use ag::tensor_ops as T;
use ag::{Context, VariableEnvironment};
use std::collections::HashMap;
use std::ops::Deref;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

struct LSTM<'g> {
    vector_dim: usize,
    hs: Vec<Tensor<'g>>,
    cells: Vec<Tensor<'g>>,
}

impl<'g> LSTM<'g> {
    /// Applies standard LSTM unit without peephole to `x`.
    /// `x` must be a tensor with shape `(batch_size, embedding_dim)`
    ///
    /// # Returns
    /// Output tensor of this unit with shape `(batch_size, state_size)`.
    fn step(
        &mut self,
        x: Tensor<'g>,
        g: &'g Context<f32>,
        var: &HashMap<&str, Tensor<'g>>,
    ) -> &Tensor<'g> {
        let (cell, h) = {
            let ref last_output = self.hs.pop().unwrap_or_else(|| T::zeros(&T::shape(x), g));
            let last_cell = self
                .cells
                .pop()
                .unwrap_or_else(|| T::zeros(&T::shape(x), g));

            let xh = T::matmul(x, var["wx"]) + T::matmul(last_output, var["wh"]) + var["b"];

            let size = self.vector_dim as isize;
            let i = T::slice(xh, &[0, 0 * size], &[-1, 1 * size]);
            let f = T::slice(xh, &[0, 1 * size], &[-1, 2 * size]);
            let c = T::slice(xh, &[0, 2 * size], &[-1, 3 * size]);
            let o = T::slice(xh, &[0, 3 * size], &[-1, 4 * size]);

            let cell = T::sigmoid(f) * last_cell + T::sigmoid(i) * T::tanh(c);
            let h = T::sigmoid(o) * T::tanh(&cell);
            (cell, h)
        };
        self.cells.push(cell);
        self.hs.push(h);
        self.hs.last().unwrap()
    }
}

fn init_vars(vocab_size: usize, vector_dim: usize, ctx: &mut VariableEnvironment<f32>) {
    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    // Embedding vector
    ctx.namespace_mut("a")
        .slot()
        .set(rng.random_normal(&[vocab_size, vector_dim], 0., 0.01));
    ctx.slot()
        .name("lookup_table")
        .set(rng.random_normal(&[vocab_size, vector_dim], 0., 0.01));
    // LSTM variable 1
    ctx.slot()
        .name("wx")
        .set(rng.random_normal(&[vector_dim, 4 * vector_dim], 0., 0.01));
    // LSTM variable 2
    ctx.slot()
        .name("wh")
        .set(rng.random_normal(&[vector_dim, 4 * vector_dim], 0., 0.01));
    // LSTM variable 3
    ctx.slot()
        .name("b")
        .set(ag::ndarray_ext::zeros(&[1, 4 * vector_dim]));
    // Variable for word prediction
    ctx.slot()
        .name("w_pred")
        .set(rng.random_uniform(&[vector_dim, vocab_size], 0., 0.01));
}

// TODO: Use real-world data
// TODO: Write in define-by-run style
pub fn main() {
    let vector_dim = 4;
    let max_sent = 3;
    let vocab_size = 5;

    let mut env = ag::VariableEnvironment::<f32>::new();
    init_vars(vocab_size, vector_dim, &mut env);

    env.run(|g| {
        let mut rnn = LSTM {
            vector_dim,
            hs: Vec::new(),
            cells: Vec::new(),
        };

        let ns = g.env().default_namespace();
        let var = g.var_tensors_by_name(&ns).collect::<HashMap<_, _>>();
        let sentences = g.placeholder("sents", &[-1, max_sent]);

        // Compute cross entropy losses for each LSTM step
        let losses: Vec<Tensor> = (0..max_sent - 1)
            .map(|i| {
                let cur_id = T::slice(sentences, &[0, i], &[-1, i + 1]);
                let next_id = T::slice(sentences, &[0, i + 1], &[-1, i + 2]);
                let x = T::squeeze(T::gather(var["lookup_table"], &cur_id, 0), &[1]);
                let h = rnn.step(x, g.deref(), &var);
                let prediction = T::matmul(h, var["w_pred"]);
                T::sparse_softmax_cross_entropy(prediction, next_id)
            })
            .collect();

        // Aggregate losses of generated words
        let loss = T::add_n(&losses);

        // Compute gradients
        let vars = &[
            var["wx"],
            var["wh"],
            var["b"],
            var["lookup_table"],
            var["w_pred"],
        ];
        let grads = T::grad(&[loss], vars);
        let mut feeder = ag::Feeder::new();
        let sents = ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]]).into_dyn();
        feeder.push(sentences, sents.view());

        // test with toy data
        ag::test_helper::check_theoretical_grads(
            loss,
            grads.as_slice(),
            vars,
            feeder,
            1e-3,
            1e-3,
            g,
        );
    });
}
