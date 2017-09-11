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
    let (((x_train, y_train), (x_test, y_test)), vocab_size) = imdb::load(max_sent_len as usize);
    let state_size = 128;
    let vec_dim = 128;
    let batch_size = 128;

    // build computation graph
    let ref tbl = ag::variable(ag::init::standard_uniform(&[vocab_size, vec_dim]));
    let ref w = ag::variable(ag::init::glorot_uniform(&[state_size, 1]));
    let ref b = ag::variable(ag::init::zeros(&[1, 1]));
    let ref sentences = ag::placeholder(&[-1, max_sent_len as isize]);
    let ref y = ag::placeholder(&[-1, 1]);
    let mut rnn = ag::nn_impl::rnn::LSTM::new(state_size, vec_dim, batch_size);

    let mut hs = vec![];
    for i in 0..max_sent_len {
        let id = ag::slice(sentences, &[0, i], &[-1, i + 1]);
        let x = ag::gather(tbl, &id, ndarray::Axis(0));
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
    let mut optimizer = ag::sgd::optimizers::Adam { ..Default::default() };
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

pub mod imdb {
    extern crate ndarray;
    extern crate glob;

    use self::glob::glob;
    use std::io;
    use std::collections::hash_map::HashMap;
    use std::process;
    use std::fs::File;
    use std::path::Path;
    use std::io::Read;
    use ndarray_ext::NdArray;

    pub fn load(max_sentence_len: usize) -> (((NdArray, NdArray), (NdArray, NdArray)), usize) {
        // load vocab
        let word2idx = load_vocab(Path::new("data/imdb/aclImdb/imdb.vocab"));
        let vocab_size = word2idx.len();

        // (25000, max_sent), (25000, max_sent,), (25000, max_sent,), (250000, max_sent,)
        let ((x_train_pos, x_train_neg), (x_test_pos, x_test_neg)) = load_x(max_sentence_len, word2idx);
        // (25000,         ), (25000,          ), (25000           ), (250000           )
        let ((y_train_pos, y_train_neg), (y_test_pos, y_test_neg)) = load_y();

        let x_train = ndarray::stack(ndarray::Axis(0), &[x_train_pos.view(), x_train_neg.view()])
            .unwrap();
        let x_test = ndarray::stack(ndarray::Axis(0), &[x_test_pos.view(), x_test_neg.view()]).unwrap();

        let y_train = ndarray::stack(ndarray::Axis(0), &[y_train_pos.view(), y_train_neg.view()])
            .unwrap();
        let y_test = ndarray::stack(ndarray::Axis(0), &[y_test_pos.view(), y_test_neg.view()]).unwrap();

        (((x_train, y_train), (x_test, y_test)), vocab_size)
    }

    fn load_x(
        max_sentence_len: usize,
        word2idx: HashMap<String, f32>,
    ) -> ((NdArray, NdArray), (NdArray, NdArray)) {

        // list of list of string
        let train_pos: Vec<Vec<String>> = load_reviews("data/imdb/aclImdb/train/pos");
        let train_neg: Vec<Vec<String>> = load_reviews("data/imdb/aclImdb/train/neg");
        let test_pos: Vec<Vec<String>> = load_reviews("data/imdb/aclImdb/test/pos");
        let test_neg: Vec<Vec<String>> = load_reviews("data/imdb/aclImdb/test/neg");

        // convert strings to ids
        let train_pos: Vec<Vec<f32>> = str2id(train_pos, &word2idx);
        let train_neg: Vec<Vec<f32>> = str2id(train_neg, &word2idx);
        let test_pos: Vec<Vec<f32>> = str2id(test_pos, &word2idx);
        let test_neg: Vec<Vec<f32>> = str2id(test_neg, &word2idx);

        // pad or truncate
        let train_pos: Vec<Vec<f32>> = align_vectors(train_pos, max_sentence_len);
        let train_neg: Vec<Vec<f32>> = align_vectors(train_neg, max_sentence_len);
        let test_pos: Vec<Vec<f32>> = align_vectors(test_pos, max_sentence_len);
        let test_neg: Vec<Vec<f32>> = align_vectors(test_neg, max_sentence_len);

        // to ndarray objects
        let train_pos: NdArray = to_tensor(train_pos, max_sentence_len);
        let train_neg: NdArray = to_tensor(train_neg, max_sentence_len);
        let test_pos: NdArray = to_tensor(test_pos, max_sentence_len);
        let test_neg: NdArray = to_tensor(test_neg, max_sentence_len);

        ((train_pos, train_neg), (test_pos, test_neg))
    }

    fn load_y() -> ((NdArray, NdArray), (NdArray, NdArray)) {
        let shape = &[12500, 1];
        let train_pos = ::initializers::ones(shape);
        let train_neg = ::initializers::zeros(shape);
        let test_pos = ::initializers::ones(shape);
        let test_neg = ::initializers::zeros(shape);

        ((train_pos, train_neg), (test_pos, test_neg))
    }

    fn align_vectors(mut vec_vec: Vec<Vec<f32>>, max_sent_len: usize) -> Vec<Vec<f32>> {
        fn align_vec(a: &mut Vec<f32>, max_sent_len: usize) {
            a.truncate(max_sent_len);
            let diff = max_sent_len - a.len();
            if diff > 0 {
                // padding
                for _ in 0..diff {
                    a.push(0.)
                }
            }
        }

        for mut vec in vec_vec.iter_mut() {
            align_vec(&mut vec, max_sent_len)
        }

        vec_vec
    }

    fn to_tensor(arg: Vec<Vec<f32>>, max_sent_len: usize) -> NdArray {
        let owned_arrays: Vec<NdArray> = arg.into_iter()
                                            .map(|vec| {
                                                NdArray::from_shape_vec(ndarray::IxDyn(&[1, vec.len()]), vec).unwrap()
                                            })
                                            .collect();

        let view_arrays: Vec<ndarray::ArrayView<f32, ndarray::IxDyn>> =
            owned_arrays.iter().map(|vec| vec.view()).collect();

        ndarray::stack(ndarray::Axis(0), view_arrays.as_slice()).unwrap()
    }

    fn str2id(a: Vec<Vec<String>>, word2idx: &HashMap<String, f32>) -> Vec<Vec<f32>> {
        a.iter()
         .map(|sent| {
             sent.iter().filter_map(|word|
                                        word2idx.get(word).map(|a| *a) // :Option<f32>
             ).collect::<Vec<f32>>()
         })
         .collect::<Vec<Vec<f32>>>()
    }

    // Returns: {word: id} mapping
    fn load_vocab(path: &Path) -> HashMap<String, f32> {
        let mut buf: Vec<u8> = vec![];
        // read into buf
        File::open(path)
            .expect("Please run ./download_imdb.sh beforehand")
            .read_to_end(&mut buf);
        // buf to string
        let whole_string = String::from_utf8_lossy(&mut buf);

        // split whole string into words (result)
        let mut result = HashMap::new();
        for w in whole_string.split("\n") {
            let id = result.len() + 2;
            result.insert(w.to_string(), id as f32);
        }
        result
    }

    fn load_reviews(path: &str) -> Vec<Vec<String>> {
        fn load_review<P: AsRef<Path>>(p: P) -> Vec<String> {
            let mut buf: Vec<u8> = vec![];
            File::open(p).expect("Please run ./download_imdb.sh beforehand").read_to_end(
                &mut buf,
            );
            let whole_string = String::from_utf8_lossy(&mut buf);

            let mut split = whole_string.split_whitespace();
            let mut words = vec![];
            while let Some(word) = split.next() {
                words.push(word.to_string());
            }
            words
        }


        let mut result = vec![];

        for entry in glob(&(path.to_string() + "/*.txt")).expect(
            "Please run ./download_imdb.sh beforehand") {
            match entry {
                Ok(p) => result.push(load_review(p)),
                Err(e) => println!("{:?}", e),
            }
        }

        result
    }

}
