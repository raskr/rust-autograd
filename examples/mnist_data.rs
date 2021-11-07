use ndarray;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::Path;

type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

/// load mnist dataset as "ndarray" objects.
///
/// Returns ((x_train, y_train), (x_test, y_test)).
///
/// Shape of x_train and x_test: (num_samples, 28)
/// Shape of y_train and y_test: (num_samples, 1)
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
    let ref mut buf_reader =
        io::BufReader::new(File::open(path).expect("Please run ./download_mnist.sh beforehand"));
    let magic = read_be_u32(buf_reader);
    if magic != 2051 {
        panic!("Invalid magic number. expected 2051, got {}", magic)
    }
    let num_image = read_be_u32(buf_reader) as usize;
    let rows = read_be_u32(buf_reader) as usize;
    let cols = read_be_u32(buf_reader) as usize;
    assert!(rows == 28 && cols == 28);

    // read images
    let mut buf: Vec<u8> = vec![0u8; num_image * rows * cols];
    let _ = buf_reader.read_exact(buf.as_mut());
    let ret = buf.into_iter().map(|x| (x as f32) / 255.).collect();
    (ret, num_image)
}

fn load_labels<P: AsRef<Path>>(path: P) -> (Vec<f32>, usize) {
    let ref mut buf_reader = io::BufReader::new(File::open(path).unwrap());
    let magic = read_be_u32(buf_reader);
    if magic != 2049 {
        panic!("Invalid magic number. Got expect 2049, got {}", magic);
    }
    let num_label = read_be_u32(buf_reader) as usize;
    // read labels
    let mut buf: Vec<u8> = vec![0u8; num_label];
    let _ = buf_reader.read_exact(buf.as_mut());
    let ret: Vec<f32> = buf.into_iter().map(|x| x as f32).collect();
    (ret, num_label)
}

// mnist is stored as big endian
fn read_be_u32<T: Read>(reader: &mut T) -> u32 {
    let mut buf: [u8; 4] = [0, 0, 0, 0];
    let _ = reader.read_exact(&mut buf);
    u32::from_be_bytes(buf)
}

#[allow(dead_code)]
fn main() {}
