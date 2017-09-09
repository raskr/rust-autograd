extern crate ndarray;
extern crate autograd as ag;

use std::time::{Duration, Instant};
use std::default::Default;


macro_rules! eval_with_time {
  ($x:expr) => {
    {
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      println!("{}.{:03} sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
      result
    }
  };
}

// Run "./download_imdb.sh" beforehand if you don't have dataset
fn main() {
    // load dataset
    let max_sent_len = 200 as isize;
    let (((x_train, y_train), (x_test, y_test)), vocab_size) =
        ag::dataset::imdb::load(max_sent_len as usize);
    let state_size = 128;
    let vec_dim = 128;
    let batch_size = 128;

    // build computation graph
    let ref tbl = ag::variable(ag::init::random_uniform(&[vocab_size, vec_dim]));
    let ref w = ag::variable(ag::init::glorot_uniform(&[state_size, 1]));
    let ref b = ag::variable(ag::init::zeros(&[1, 1]));
    let ref sentences = ag::placeholder(&[-1, max_sent_len as isize]);
    let ref y = ag::placeholder(&[-1, 1]);
    let mut rnn = ag::nn_impl::rnn::LSTM::new(state_size, vec_dim, batch_size);

    let mut hs = vec![];
    for i in 0..max_sent_len {
        let id = ag::slice(sentences, &[0, i], &[-1, i + 1]);
        let x = ag::embedding_lookup(tbl, &id);
        let h = ag::rnn_step(&x, &mut rnn, i==max_sent_len-1);
        hs.push(h);
    }
    let ref last_h = hs.last().unwrap();
    let ref logits = ag::matmul(last_h, w) + b;
    let ref loss = ag::sigmoid_cross_entropy(logits, y);
    let mut params = rnn.list_vars();
    params.extend_from_slice(&[tbl, w]);
    let ref grads = ag::gradients(loss, params.as_slice(), None);

    // training
    let mut optimizer = ag::sgd::Adam { ..Default::default() };
    let num_samples = x_train.shape()[0];
    let num_batches = num_samples / batch_size;

    for epoch in 0..5 {
        let perm = ag::init::permutation(num_samples).to_vec();
        for i in 0..num_batches {
            let indices = perm[i..i+batch_size].to_vec();
            let x_batch = x_train.select(ndarray::Axis(0), indices.as_slice());
            let y_batch = y_train.select(ndarray::Axis(0), indices.as_slice());
            let feed_dict = ag::Input::new().add(sentences, x_batch).add(y, y_batch);
            // uagate shared variables
            ag::train::apply_gradients(&mut optimizer, params.as_slice(), grads, feed_dict);
        }
        println!("finished epoch {}", epoch);
    }

}
