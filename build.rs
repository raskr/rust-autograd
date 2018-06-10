extern crate cc;

fn main()
{
    cc::Build::new()
        .flag("-std=c99")
        .file("src/c/conv.c")
        .compile("libconv.a");
}
