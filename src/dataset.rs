pub mod mnist {
    extern crate ndarray;

    use std::mem;
    use std::io;
    use std::collections::hash_map::HashMap;
    use std::process;
    use std::fs::File;
    use std::path::Path;
    use std::io::Read;
    use ndarray_ext::NdArray;

    /// load mnist dataset as "ndarray" objects.
    ///
    /// labels are sparse (vertical vector).
    pub fn load() -> ((NdArray, NdArray), (NdArray, NdArray)) {
        // load dataset as `Vec`s
        let (train_x, num_image_train): (Vec<f32>, usize) =
            load_images("data/mnist/train-images-idx3-ubyte");
        let (train_y, num_label_train): (Vec<f32>, usize) =
            load_labels("data/mnist/train-labels-idx1-ubyte");
        let (test_x, num_image_test): (Vec<f32>, usize) =
            load_images("data/mnist/t10k-images-idx3-ubyte");
        let (test_y, num_label_test): (Vec<f32>, usize) =
            load_labels("data/mnist/t10k-labels-idx1-ubyte");

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        let x_train = as_arr(ndarray::IxDyn(&[num_image_train, 28 * 28]), train_x).unwrap();
        let y_train = as_arr(ndarray::IxDyn(&[num_label_train, 1]), train_y).unwrap();
        let x_test = as_arr(ndarray::IxDyn(&[num_image_test, 28 * 28]), test_x).unwrap();
        let y_test = as_arr(ndarray::IxDyn(&[num_label_test, 1]), test_y).unwrap();
        ((x_train, y_train), (x_test, y_test))
    }

    fn load_images<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
        let ref mut buf_reader = io::BufReader::new(File::open(path).expect(
            "Please run ./download_mnist.sh beforehand"));

        let magic = u32::from_be(read_u32(buf_reader));

        if magic != 2051 {
            panic!("Invalid magic number. expected 2051, got {}", magic)
        }

        let num_image = u32::from_be(read_u32(buf_reader)) as usize;
        let rows = u32::from_be(read_u32(buf_reader)) as usize;
        let cols = u32::from_be(read_u32(buf_reader)) as usize;
        assert!(rows == 28 && cols == 28);

        // read images
        let mut buf: Vec<u8> = vec![0u8; num_image * rows * cols];
        buf_reader.read_exact(buf.as_mut());
        let ret = buf.into_iter().map(|x| (x as f32) / 255.).collect();
        (ret, num_image)
    }

    fn load_labels<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
        let ref mut buf_reader = io::BufReader::new(File::open(path).unwrap());

        let magic = u32::from_be(read_u32(buf_reader));

        if magic != 2049 {
            panic!("Invalid magic number. Got expect 2049, got {}", magic);
        }

        let num_label = u32::from_be(read_u32(buf_reader)) as usize;

        // read labels
        let mut buf: Vec<u8> = vec![0u8; num_label];
        buf_reader.read_exact(buf.as_mut());
        let ret: Vec<f32> = buf.into_iter().map(|x| x as f32).collect();
        (ret, num_label)
    }

    fn read_u32<T: Read>(reader: &mut T) -> u32 {
        let mut buf: [u8; 4] = [0, 0, 0, 0];
        reader.read_exact(&mut buf);
        unsafe { mem::transmute(buf) }
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


